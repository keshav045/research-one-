"""
Unit tests for ResearchLens LangGraph routing and state machine nodes.
Tests conditional edge routing, preflight decisions, and bounded retry loops.
"""

from __future__ import annotations

import pytest
from langgraph.graph import END

from backend.services.research_graph import (
    ResearchState,
    preflight_node,
    route_after_preflight,
    route_after_verify,
    rewrite_search_queries,
    research_graph,
    MIN_VERIFIED_FACTUAL,
    MIN_VERIFIED_REVIEW,
    MAX_RETRIES,
)


def test_rejected_query_ends():
    """Verify that rejected queries route directly to END without invoking planner or retrieval."""
    state: ResearchState = {"allowed": False, "question": "too short"}
    destination = route_after_preflight(state)
    assert destination == END


def test_allowed_query_proceeds_to_planner():
    """Verify that allowed queries proceed to the plan_query node."""
    state: ResearchState = {"allowed": True, "question": "What is the Transformer architecture?"}
    destination = route_after_preflight(state)
    assert destination == "plan_query"


def test_route_after_verify_retry_when_empty():
    """If verified claims are below threshold and retries remain, route to rewrite_query."""
    state: ResearchState = {
        "intent": "literature_review",
        "verified": [],
        "retries": 0,
        "max_retries": 2,
    }
    assert route_after_verify(state) == "rewrite_query"


def test_route_after_verify_stops_at_max_retries():
    """If retries reach MAX_RETRIES, route to write_report even if verified claims are below threshold."""
    state: ResearchState = {
        "intent": "literature_review",
        "verified": [],
        "retries": 2,
        "max_retries": 2,
    }
    assert route_after_verify(state) == "write_report"


def test_route_after_verify_stops_when_threshold_met():
    """If verified claims meet or exceed the threshold, proceed to write_report on attempt 0."""
    state: ResearchState = {
        "intent": "literature_review",
        "verified": ["claim1", "claim2", "claim3"],  # type: ignore[list-item]
        "retries": 0,
        "max_retries": 2,
    }
    assert route_after_verify(state) == "write_report"


def test_route_after_verify_factual_lookup():
    """Factual lookup requires at least 1 verified claim to proceed."""
    state_empty: ResearchState = {
        "intent": "factual_lookup",
        "verified": [],
        "retries": 0,
        "max_retries": 2,
    }
    assert route_after_verify(state_empty) == "rewrite_query"

    state_one: ResearchState = {
        "intent": "factual_lookup",
        "verified": ["claim1"],  # type: ignore[list-item]
        "retries": 0,
        "max_retries": 2,
    }
    assert route_after_verify(state_one) == "write_report"


def test_preflight_node_behavior():
    """Test preflight node with valid and invalid inputs."""
    # Under minimum 10 chars
    res_short = preflight_node({"question": "tiny"})
    assert res_short["allowed"] is False
    assert "Minimum question length is 10 characters" in res_short["reject_reason"]

    # Valid academic question
    res_valid = preflight_node({"question": "Which paper introduced the Transformer architecture?"})
    assert res_valid["allowed"] is True
    assert res_valid["reject_reason"] == ""
    assert res_valid["retries"] == 0


@pytest.mark.asyncio
async def test_rewrite_search_queries_fallback():
    """Verify that rewrite_search_queries produces distinct queries without crashing."""
    question = "Which paper introduced the Transformer architecture and self-attention?"
    prev = ["Transformer architecture self-attention"]
    new_queries = await rewrite_search_queries(
        question=question,
        previous_queries=prev,
        rejected_claims=["No passage mentioned Transformer"],
    )
    assert len(new_queries) >= 1
    # Check that it doesn't just duplicate previous query
    assert new_queries != prev


def test_graph_has_required_nodes():
    """Verify that all nodes and edges are compiled in the state graph."""
    node_keys = research_graph.nodes.keys()
    for expected in ["preflight", "plan_query", "retrieve", "rerank_fetch", "verify_claims", "rewrite_query", "write_report"]:
        assert expected in node_keys


@pytest.mark.asyncio
async def test_graph_retry_loop_execution(monkeypatch):
    """
    Test the full graph execution loop with monkeypatched nodes.
    Simulate verify_claims failing on attempt 0 and 1, then passing on attempt 2.
    Confirm that the retry counter reaches 2 and write_report is reached.
    """
    verify_call_count = 0
    rewrite_call_count = 0

    async def mock_plan_query(state):
        return {
            "intent": "literature_review",
            "sub_queries": [state["question"]],
            "expected_titles": [],
            "sub_questions": [state["question"]],
        }

    async def mock_retrieve(state):
        return {"papers": []}

    async def mock_rerank_fetch(state):
        return {"ranked_papers": [], "passages_total": 5}

    async def mock_verify_claims(state):
        nonlocal verify_call_count
        verify_call_count += 1
        # Fail on tries 1 and 2, pass on try 3
        if verify_call_count < 3:
            return {"verified": [], "rejected": ["insufficient empirical data"]}
        return {"verified": ["verified_citation_1", "verified_citation_2"], "rejected": []}

    async def mock_rewrite_query(state):
        nonlocal rewrite_call_count
        rewrite_call_count += 1
        current_retries = state.get("retries", 0)
        return {
            "sub_queries": [f"query rewrite {rewrite_call_count}"],
            "retries": current_retries + 1,
            "retry_history": state.get("retry_history", []) + [{"retry": current_retries + 1}],
        }

    async def mock_write_report(state):
        from backend.models.schemas import ResearchReport
        return {
            "report": ResearchReport(
                executiveSummary="Mock summary",
                methodology="Mock methodology",
                findings=[],
                comparisonTable=[],
                computationalRequirements="",
                contradictoryEvidence="",
                limitations=[],
                conclusion="Mock conclusion",
                references=[],
            ),
            "limited_evidence": False,
            "status": "completed",
        }

    from backend.services import research_graph as rg
    monkeypatch.setattr(rg, "plan_query_node", mock_plan_query)
    monkeypatch.setattr(rg, "retrieve_node", mock_retrieve)
    monkeypatch.setattr(rg, "rerank_fetch_node", mock_rerank_fetch)
    monkeypatch.setattr(rg, "verify_claims_node", mock_verify_claims)
    monkeypatch.setattr(rg, "rewrite_query_node", mock_rewrite_query)
    monkeypatch.setattr(rg, "write_report_node", mock_write_report)

    test_graph = rg.build_research_graph()
    result = await test_graph.ainvoke({"question": "How do Transformers improve over recurrent models?"})

    assert result["allowed"] is True
    assert verify_call_count == 3
    assert rewrite_call_count == 2
    assert result["retries"] == 2
    assert result["status"] == "completed"
    assert result["report"] is not None

