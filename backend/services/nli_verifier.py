"""
NLI Claim Verification Service — PyTorch & DeBERTa
==================================================
Uses MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli to verify whether
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
from typing import Any, Optional, List, Tuple

import torch
if not hasattr(torch, "accelerator"):
    class _DummyAccelerator:
        @staticmethod
        def current_accelerator():
            return None
    torch.accelerator = _DummyAccelerator()

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ..models.schemas import AtomicClaimVerification, EntailmentVerdict
from ..config import settings
from .text_utils import split_into_sentences

logger = logging.getLogger(__name__)

_tokenizer = None
_model = None
_device = None

# Typical id2label for DeBERTa NLI: {0: 'contradiction', 1: 'entailment', 2: 'neutral'}


def get_nli_components():
    """Lazy-load the tokenizer and sequence classification model on CUDA/CPU."""
    global _tokenizer, _model, _device
    if _model is None or _tokenizer is None:
        model_name = settings.NLI_MODEL or "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
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

    nums_in_claim = re.findall(r"\b\d+(?:,\d+)*(?:\.\d+)?(?:x|%|gb|mb|fps|bits?)?\b", c_lower)
    p_lower_clean = p_lower.replace(",", "")
    nums_supported = all(n.replace(",", "") in p_lower_clean for n in nums_in_claim)

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

    # Always return NEUTRAL on heuristic fallback to prevent false positives (never ENTAILS)
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


# ─── Step 6B: Answer-Level NLI Verification ───────────────────────────────────

def _extract_premise_for_citation(cit: Any) -> str:
    """Extract cited claim sentence plus one neighboring sentence on each side from source passage."""
    passage = getattr(cit, "passage", "") or ""
    if not passage:
        return getattr(cit, "claim", "") or getattr(cit, "highlightSentence", "") or ""

    sents = split_into_sentences(passage)
    if not sents:
        return passage

    target = (getattr(cit, "highlightSentence", "") or getattr(cit, "claim", "") or "").strip().lower()
    best_idx = 0
    best_overlap = -1
    target_words = set(re.findall(r"\w+", target))

    for idx, s in enumerate(sents):
        s_words = set(re.findall(r"\w+", s.lower()))
        overlap = len(target_words & s_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = idx

    start_idx = max(0, best_idx - 1)
    end_idx = min(len(sents), best_idx + 2)
    return " ".join(sents[start_idx:end_idx])


def verify_answer_sentences(
    answer_text: str,
    citations: list[Any],
    anchor_paper: Optional[Any] = None,
    first_sentence: Optional[str] = None,
) -> tuple[str, float, list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Step 6B Answer-level NLI:
    - First sentence (title, authors, year) is marked type=metadata, not run through NLI,
      and not counted in the integrity denominator.
    - Evaluates substantive claim sentences against cited passage premises.
    - Integrity = verified claim sentences / claim sentences.
    - Returns (verified_answer_text, integrity_ratio, verified_details, removed_details).
    """
    if not answer_text or not answer_text.strip():
        if first_sentence:
            return first_sentence, 0.0, [{"sentence": first_sentence, "type": "metadata", "status": "retained"}], []
        return "", 0.0, [], []

    badge_map = {getattr(c, "badgeNumber", 0): c for c in citations}
    raw_sentences = split_into_sentences(answer_text)
    if not raw_sentences:
        if first_sentence:
            return first_sentence, 0.0, [{"sentence": first_sentence, "type": "metadata", "status": "retained"}], []
        return "", 0.0, [], []

    # Step 2a: Identify the first sentence (title, authors, year)
    metadata_sentence = None
    if first_sentence:
        if raw_sentences and raw_sentences[0].strip() == first_sentence.strip():
            metadata_sentence = raw_sentences.pop(0)
        else:
            metadata_sentence = first_sentence.strip()
    elif raw_sentences:
        first = raw_sentences[0]
        # Detect if first sentence is introductory metadata without citations
        if not re.search(r"\[\d+\]", first) and any(
            kw in first.lower() for kw in ("introduced by", "proposed by", "was introduced", "introduced the", "introduces the", "no definitive foundational")
        ):
            metadata_sentence = raw_sentences.pop(0)

    # Claim sentences are the substantive assertion sentences (denominator of integrity)
    claim_sentences = list(raw_sentences)

    threshold = getattr(settings, "NLI_ENTAIL_THRESHOLD", 0.80)
    pairs_to_eval: list[tuple[str, str, str, list[int]]] = []

    for sent in claim_sentences:
        badges = [int(m) for m in re.findall(r"\[(\d+)\]", sent)]
        if badges:
            premise_parts = []
            for b in badges:
                c = badge_map.get(b)
                if c:
                    premise_parts.append(_extract_premise_for_citation(c))
            premise_text = " ".join(premise_parts) if premise_parts else ""
        else:
            premise_text = ""

        clean_hypothesis = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", sent).strip()
        pairs_to_eval.append((premise_text, clean_hypothesis, sent, badges))

    verified_claim_sentences: list[str] = []
    verified_details: list[dict[str, Any]] = []
    removed_details: list[dict[str, Any]] = []

    if metadata_sentence:
        verified_details.append({
            "sentence": metadata_sentence,
            "type": "metadata",
            "status": "retained",
        })

    for premise, hypothesis, original_sent, badges in pairs_to_eval:
        if not premise:
            removed_details.append({
                "sentence": original_sent,
                "type": "claim",
                "reason": "Missing supporting citation badges or ground-truth premise",
                "entailment_score": 0.0,
                "verdict": "UNSUPPORTED",
            })
            logger.info("[NLI-Answer] Removed uncited/unsupported claim sentence: '%s'", original_sent[:80])
            continue

        try:
            res = _evaluate_batch_pairs([(premise, hypothesis)])
            verdict, confidence, reasoning = res[0]
            entail_score = confidence if verdict == EntailmentVerdict.ENTAILS else (0.50 if verdict == EntailmentVerdict.NEUTRAL else 0.0)
        except Exception as exc:
            logger.warning("[NLI-Answer] Inference failed for sentence, using heuristic fallback: %s", exc)
            verdict, confidence, reasoning = _heuristic_entailment(premise, hypothesis, premise)
            entail_score = 0.50 if verdict == EntailmentVerdict.NEUTRAL else 0.0

        if verdict == EntailmentVerdict.ENTAILS and confidence >= threshold:
            verified_claim_sentences.append(original_sent)
            verified_details.append({
                "sentence": original_sent,
                "type": "claim",
                "entailment_score": confidence,
                "verdict": "VERIFIED",
                "badges": badges,
                "reasoning": reasoning,
            })
        else:
            removed_details.append({
                "sentence": original_sent,
                "type": "claim",
                "reason": f"Entailment score {confidence:.3f} below threshold {threshold:.2f} ({verdict.value if hasattr(verdict, 'value') else verdict})",
                "entailment_score": confidence,
                "verdict": verdict.value if hasattr(verdict, "value") else str(verdict),
                "reasoning": reasoning,
            })
            logger.info(
                "[NLI-Answer] Removed unsupported sentence: '%s' (verdict=%s, conf=%.3f)",
                original_sent[:80], verdict, confidence
            )

    # Integrity = verified claim sentences / claim sentences
    total_claim_sentences = len(claim_sentences)
    integrity = (len(verified_claim_sentences) / total_claim_sentences) if total_claim_sentences > 0 else 0.0

    if metadata_sentence and verified_claim_sentences:
        verified_answer_text = f"{metadata_sentence} " + " ".join(verified_claim_sentences)
    elif metadata_sentence:
        verified_answer_text = metadata_sentence
    else:
        verified_answer_text = " ".join(verified_claim_sentences)

    return verified_answer_text, round(integrity, 3), verified_details, removed_details
