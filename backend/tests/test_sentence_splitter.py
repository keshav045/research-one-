"""
Unit tests for abbreviation-aware sentence splitter.
Tests:
- Author initials: 'Diederik P. Kingma', 'Jimmy L. Ba'
- Academic abbreviations: 'et al.', 'Fig.', 'Eq.', 'Sec.', 'No.', 'vs.', 'e.g.', 'i.e.'
- Numbers with decimals
- Citation markers
"""

import pytest
from backend.services.text_utils import split_into_sentences


def test_author_initials_kingma():
    text = (
        "Adam was introduced by Diederik P. Kingma and Jimmy Ba in 2014 [1]. "
        "The method computes individual adaptive learning rates for different parameters [1]."
    )
    sents = split_into_sentences(text)
    assert len(sents) == 2, f"Expected 2 sentences, got {len(sents)}: {sents}"
    assert "Diederik P. Kingma" in sents[0]
    assert sents[0] == "Adam was introduced by Diederik P. Kingma and Jimmy Ba in 2014 [1]."
    assert sents[1] == "The method computes individual adaptive learning rates for different parameters [1]."


def test_abbreviations():
    text = (
        "Vaswani et al. introduced the Transformer [1]. "
        "See Fig. 2 and Eq. 4 in Sec. 3 for architectural details. "
        "Model A vs. Model B showed superior performance, e.g. on WMT 2014, i.e. achieving 28.4 BLEU. "
        "Experiment No. 1 evaluated 3.14 learning rate decay."
    )
    sents = split_into_sentences(text)
    assert len(sents) == 4, f"Expected 4 sentences, got {len(sents)}: {sents}"
    assert sents[0].startswith("Vaswani et al.")
    assert "Fig. 2 and Eq. 4 in Sec. 3" in sents[1]
    assert "Model A vs. Model B" in sents[2]
    assert "e.g. on WMT 2014, i.e. achieving 28.4 BLEU." in sents[2]
    assert "Experiment No. 1" in sents[3]
    assert "3.14" in sents[3]


def test_empty_and_single_sentence():
    assert split_into_sentences("") == []
    assert split_into_sentences("   ") == []
    assert split_into_sentences("Single sentence without punctuation") == ["Single sentence without punctuation"]
    assert split_into_sentences("Single sentence with punctuation.") == ["Single sentence with punctuation."]
