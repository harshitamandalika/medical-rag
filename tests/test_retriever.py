import pytest
from retrieval.sparse_retriever import _matches_filter


def test_matches_filter_plain_equality():
    meta = {"condition": "type2_diabetes", "year": 2024}
    assert _matches_filter(meta, {"condition": "type2_diabetes"}) is True
    assert _matches_filter(meta, {"condition": "hypertension"}) is False


def test_matches_filter_chroma_eq():
    meta = {"condition": "type2_diabetes"}
    assert _matches_filter(meta, {"condition": {"$eq": "type2_diabetes"}}) is True
    assert _matches_filter(meta, {"condition": {"$eq": "asthma"}}) is False


def test_matches_filter_chroma_gte():
    meta = {"year": 2022}
    assert _matches_filter(meta, {"year": {"$gte": 2020}}) is True
    assert _matches_filter(meta, {"year": {"$gte": 2023}}) is False


def test_matches_filter_multiple_conditions():
    meta = {"condition": "type2_diabetes", "year": 2023}
    assert _matches_filter(meta, {
        "condition": {"$eq": "type2_diabetes"},
        "year": {"$gte": 2020},
    }) is True