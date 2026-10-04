"""
NLI Claim Verification Service — PyTorch & DeBERTa
==================================================
Uses cross-encoder/nli-deberta-v3-small to verify whether
a source passage (premise) entails, contradicts, or is neutral toward
each atomic claim (hypothesis).

Uses AutoTokenizer & AutoModelForSequenceClassification directly to:
1. Avoid Windows Application Control / sklearn Cython DLL blocks.
2. Run on CUDA with batching for high throughput.
3. Enforce strict thresholds per Phase 5.6 of the architecture plan.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Optional, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ..models.schemas import AtomicClaimVerification, EntailmentVerdict
from ..config import settings

logger = logging.getLogger(__name__)

_tokenizer = None
_model = None
_device = None

# Typical id2label for cross-encoder/nli-deberta-v3-small: {0: 'contradiction', 1: 'entailment', 2: 'neutral'}


def get_nli_components():
    """Lazy-load the tokenizer and sequence classification model on CUDA/CPU."""
    global _tokenizer, _model, _device
    if _model is None or _tokenizer is None:
        model_name = settings.NLI_MODEL or "cross-encoder/nli-deberta-v3-small"
        logger.info("[NLI] Loading model and tokenizer: %s", model_name)
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        target_device = settings.NLI_DEVICE if hasattr(settings, "NLI_DEVICE") else "cuda"
        if target_device == "cuda" and torch.cuda.is_available():
            _device = torch.device("cuda:0")
            logger.info("[NLI] Model loaded on CUDA:0")
        else:
            _device = torch.device("cpu")
            logger.info("[NLI] Model loaded on CPU")
            
        _model.to(_device)
        _model.eval()
    return _tokenizer, _model, _device


# ─── Claim Decomposition ──────────────────────────────────────────────────────

_SPLIT_PATTERNS = [
    re.compile(r"\s*;\s*"),
    re.compile(r"\s*,\s*and\s+(?:also\s+)?", re.IGNORECASE),
    re.compile(r"\s*,\s*while\s+(?:simultaneously\s+)?", re.IGNORECASE),
    re.compile(r"\s*,\s*with\s+(?:only\s+)?", re.IGNORECASE),
    re.compile(r"\s*,\s*as well as\s+", re.IGNORECASE),
    re.compile(r"\s*,\s*resulting in\s+", re.IGNORECASE),
    re.compile(r"\s*,\s*leading to\s+", re.IGNORECASE),
    re.compile(r"\s*,\s*whereas\s+", re.IGNORECASE),
]


def decompose_claim(claim: str) -> list[str]:
    """Split a compound claim into independent atomic assertions."""
    if not claim:
        return []

    clean = re.sub(r"\s+", " ", claim.strip())
    parts = [clean]

    for pattern in _SPLIT_PATTERNS:
        next_parts: list[str] = []
        for part in parts:
            split = pattern.split(part)
            next_parts.extend(s.strip() for s in split if s.strip())
        parts = next_parts

    if len(parts) == 1 and len(parts[0]) > 60:
        mid_and = re.split(r"\s+and\s+(?=[a-z]+\s+(?:by|to|with|on|in)\b)", parts[0], flags=re.IGNORECASE)
        if len(mid_and) > 1:
            parts = mid_and

    result: list[str] = []
    seen: set[str] = set()
    for part in parts:
        part = part.strip().lstrip("and ").lstrip("while ").lstrip("with ")
        if not part:
            continue
        part = part[0].upper() + part[1:]
        if not part.endswith("."):
            part += "."
        if part not in seen:
            seen.add(part)
            result.append(part)

    result = [p for p in result if len(p.split()) >= 5]
    return result or [clean if clean.endswith(".") else clean + "."]


# ─── NLI Model Inference ──────────────────────────────────────────────────────

def _evaluate_batch_pairs(pairs: List[Tuple[str, str]]) -> List[Tuple[EntailmentVerdict, float, str]]:
    """
    Run NLI inference on a list of (premise, hypothesis) pairs in batches.
    Returns (verdict, confidence, reasoning) for each pair.
    """
    if not pairs:
        return []

    tokenizer, model, device = get_nli_components()
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    
    batch_size = 16
    results = []

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        premises = [p[0] for p in batch]
        hypotheses = [p[1] for p in batch]

        inputs = tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu()

        for j in range(len(batch)):
            row_probs = probs[j]
            label_scores: dict[str, float] = {}
            for idx, label_name in id2label.items():
                if "entail" in label_name:
                    label_scores["entailment"] = float(row_probs[idx])
                elif "contradict" in label_name:
                    label_scores["contradiction"] = float(row_probs[idx])
                else:
                    label_scores["neutral"] = float(row_probs[idx])

            entail_prob = label_scores.get("entailment", 0.0)
            contra_prob = label_scores.get("contradiction", 0.0)
            neutral_prob = label_scores.get("neutral", 0.0)

            # Strict Phase 5.6 thresholds
            if entail_prob >= 0.80:
                verdict = EntailmentVerdict.ENTAILS
                confidence = entail_prob
                reasoning = (
                    f"Fully entailed: The source passage directly supports this claim "
                    f"with high confidence (entailment {entail_prob:.1%})."
                )
            elif contra_prob >= 0.70:
                verdict = EntailmentVerdict.CONTRADICTS
                confidence = contra_prob
                reasoning = (
                    f"Contradiction detected: The source passage conflicts with this claim "
                    f"(contradiction {contra_prob:.1%})."
                )
            elif entail_prob >= 0.50:
                verdict = EntailmentVerdict.NEUTRAL
                confidence = entail_prob
                reasoning = (
                    f"Partially supported: The passage aligns with the assertion but lacks complete "
                    f"entailing evidence (entailment {entail_prob:.1%}, neutral {neutral_prob:.1%})."
                )
            else:
                verdict = EntailmentVerdict.NEUTRAL
                confidence = max(neutral_prob, entail_prob)
                reasoning = (
                    f"Unsupported: Insufficient evidence in passage to confirm assertion "
                    f"(neutral {neutral_prob:.1%}, entailment {entail_prob:.1%})."
                )

            results.append((verdict, round(confidence, 3), reasoning))

    return results


def _heuristic_entailment(
    premise: str,
    claim: str,
    matched_sentence: str,
) -> tuple[EntailmentVerdict, float, str]:
    """
    Directional heuristic fallback used only if NLI model fails to load.
    CRITICAL RULE (Phase 5.6): Must return NEUTRAL by default, NEVER ENTAILS.
    """
    p_lower = premise.lower()
    c_lower = claim.lower()
    s_lower = matched_sentence.lower()

    nums_in_claim = re.findall(r"\b\d+(?:\.\d+)?(?:x|%|gb|mb|fps|bits?)?\b", c_lower)
    nums_supported = all(n in p_lower for n in nums_in_claim)

    is_claim_pos = bool(re.search(r"\b(improves|increases|speedup|accelerates|higher|outperforms)\b", c_lower))
    is_claim_neg = bool(re.search(r"\b(reduces|decreases|diminishes|degrades|lower)\b", c_lower))
    is_prem_neg  = bool(re.search(r"\b(diminish|plateau|degrade|limit|loss)\b", s_lower))
    is_prem_pos  = bool(re.search(r"\b(improve|increase|speedup|superior|outperform)\b", s_lower))

    if (is_claim_pos and is_prem_neg and not is_prem_pos) or (is_claim_neg and is_prem_pos and not is_prem_neg):
        return EntailmentVerdict.CONTRADICTS, 0.75, "Contradiction indicated by directional keyword conflict."
    
    if nums_in_claim and not nums_supported:
        return EntailmentVerdict.NEUTRAL, 0.60, (
            f"Neutral: numbers ({', '.join(nums_in_claim)}) not confirmed verbatim in passage."
        )

    # Always return NEUTRAL on heuristic fallback to prevent false positives
    return EntailmentVerdict.NEUTRAL, 0.50, "Neutral: heuristic fallback cannot guarantee full entailment."


def verify_atomic_claim(
    premise: str,
    atomic_claim: str,
    matched_sentence: str = "",
) -> AtomicClaimVerification:
    """Verify one atomic claim against a passage."""
    try:
        res = _evaluate_batch_pairs([(premise, atomic_claim)])
        verdict, confidence, reasoning = res[0]
    except Exception as exc:
        logger.warning("[NLI] Model inference failed, falling back to heuristic: %s", exc)
        verdict, confidence, reasoning = _heuristic_entailment(premise, atomic_claim, matched_sentence or premise)

    return AtomicClaimVerification(
        id=f"atomic-{uuid.uuid4().hex[:8]}",
        atomicClaim=atomic_claim,
        matchedSentence=matched_sentence or premise[:120],
        verdict=verdict,
        confidence=confidence,
        reasoning=reasoning,
    )


def verify_claims_batch(
    pairs: list[tuple[str, str, str]],  # (premise, atomic_claim, matched_sentence)
) -> list[AtomicClaimVerification]:
    """Verify multiple (premise, claim, sentence) triples efficiently."""
    if not pairs:
        return []

    try:
        eval_pairs = [(p[0], p[1]) for p in pairs]
        results = _evaluate_batch_pairs(eval_pairs)
        
        verifications = []
        for i, (verdict, confidence, reasoning) in enumerate(results):
            premise, claim, matched = pairs[i]
            verifications.append(
                AtomicClaimVerification(
                    id=f"atomic-{uuid.uuid4().hex[:8]}",
                    atomicClaim=claim,
                    matchedSentence=matched or premise[:120],
                    verdict=verdict,
                    confidence=confidence,
                    reasoning=reasoning,
                )
            )
        return verifications
    except Exception as exc:
        logger.warning("[NLI] Batch inference failed, falling back to individual heuristic: %s", exc)
        verifications = []
        for premise, claim, matched in pairs:
            verdict, confidence, reasoning = _heuristic_entailment(premise, claim, matched)
            verifications.append(
                AtomicClaimVerification(
                    id=f"atomic-{uuid.uuid4().hex[:8]}",
                    atomicClaim=claim,
                    matchedSentence=matched or premise[:120],
                    verdict=verdict,
                    confidence=confidence,
                    reasoning=reasoning,
                )
            )
        return verifications
