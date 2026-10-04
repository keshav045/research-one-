"""
Tests for STEP 6: Data, Fallback Readers, Early Returns, and UI
================================================================
Verifies:
a) authors_json and passages_json tolerate apostrophes in author names and passage text.
   Fallback reader tolerates old str(...) format.
b) extract_and_verify_evidence returns 4-tuple on empty papers list.
c) decompose_claim is deleted from nli_verifier.
"""

import json
import pytest
from backend.models.database import PaperRecord, SessionLocal, engine, Base
from backend.services.evidence import extract_and_verify_evidence
import backend.services.nli_verifier as nli_v


def test_authors_and_passages_with_apostrophes():
    """Verify json.dumps stores and get_authors / get_passages reads apostrophes accurately."""
    authors = ["Terence O'Reilly", "Liam D'Angelo"]
    passages = [
        {"id": "p-1", "page": 1, "section": "intro", "text": "We can't ignore parallelization; it doesn't require recurrence."}
    ]

    # Save using json.dumps
    rec = PaperRecord(
        id="job1::p-1",
        job_id="job1",
        title="Modern Transformers",
        authors_json=json.dumps(authors),
        passages_json=json.dumps(passages),
    )

    read_authors = rec.get_authors()
    read_passages = rec.get_passages()

    assert read_authors == ["Terence O'Reilly", "Liam D'Angelo"]
    assert len(read_passages) == 1
    assert "can't" in read_passages[0]["text"]
    assert "doesn't" in read_passages[0]["text"]


def test_fallback_migration_reader_tolerates_old_format():
    """Verify get_authors and get_passages tolerate old str(list) format."""
    # Old python list repr
    old_authors_repr = "['Terence O\\'Reilly', 'Liam D\\'Angelo']"
    old_passages_repr = "[{'id': 'p-1', 'page': 1, 'section': 'intro', 'text': 'We can\\'t wait'}]"

    rec = PaperRecord(
        id="job1::p-old",
        job_id="job1",
        title="Old Format Paper",
        authors_json=old_authors_repr,
        passages_json=old_passages_repr,
    )

    read_authors = rec.get_authors()
    read_passages = rec.get_passages()

    assert "Terence O'Reilly" in read_authors
    assert len(read_passages) == 1
    assert read_passages[0]["id"] == "p-1"


def test_evidence_early_return_matches_4_tuple():
    """When papers list is empty, extract_and_verify_evidence returns a 4-tuple."""
    res = extract_and_verify_evidence(
        question="What is Transformer?",
        sub_questions=["What is Transformer?"],
        papers=[],
        vector_store=None,
    )
    assert isinstance(res, tuple)
    assert len(res) == 4
    claims, citations, rejected, removals = res
    assert claims == []
    assert citations == []
    assert rejected == []
    assert isinstance(removals, dict)


def test_decompose_claim_deleted():
    """decompose_claim must be removed from nli_verifier."""
    assert not hasattr(nli_v, "decompose_claim")
