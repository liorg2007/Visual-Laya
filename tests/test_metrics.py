import math

import pytest

from laya_vision.eval.metrics import answer_probs, compute_metrics, markdown_table, reliability_table, summarize


def _rows():
    return [
        {"id": "a", "task": "t1", "modality": "image", "qtype": "choice", "probs": [0.7, 0.2, 0.1], "target": [1, 0, 0]},
        {"id": "b", "task": "t2", "modality": "text", "qtype": "choice", "probs": [0.4, 0.6], "target": [1, 0]},
    ]


def test_summary_hand_computed():
    m = summarize(_rows())
    assert m["n"] == 2 and m["skipped"] == 0
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["brier"] == pytest.approx((0.14 + 0.72) / 2)
    assert m["nll"] == pytest.approx((-math.log(0.7) - math.log(0.4)) / 2)
    # conf 0.7 (correct) and 0.6 (wrong) land in different bins: 0.5*0.3 + 0.5*0.6
    assert m["ece10"] == pytest.approx(0.45)
    assert m["ece15"] == pytest.approx(0.45)
    assert "score_mae" not in m


def test_score_mae_soft_target_and_skipped():
    rows = [{"id": "s", "task": "k", "modality": "image", "qtype": "score", "probs": [0.2, 0.3, 0.5], "target": [0, 1, 0]},
            {"id": "x", "task": "k", "modality": "image", "qtype": "choice", "probs": None, "target": [1, 0]}]
    m = summarize(rows)
    assert m["n"] == 1 and m["skipped"] == 1
    assert m["score_mae"] == pytest.approx(0.3)
    # soft target: gold level is its expectation (0.5*1 + 0.5*2 = 1.5)
    rows[0]["target"] = [0, 0.5, 0.5]
    assert summarize(rows[:1])["score_mae"] == pytest.approx(abs(1.3 - 1.5))


def test_reliability_table_counts():
    t = reliability_table([0.05, 0.6, 0.7, 1.0], [0, 0, 1, 1], bins=10)
    assert [b["count"] for b in t] == [1, 0, 0, 0, 0, 1, 1, 0, 0, 1]
    assert t[9]["accuracy"] == 1.0 and t[1]["confidence"] is None


def test_answer_probs():
    assert answer_probs({"type": "noul", "noul": 0.8}, {"type": "noul"}) == pytest.approx([0.2, 0.8])
    assert answer_probs({"type": "score", "probabilities": {"0": 0.1, "1": 0.9}},
                        {"type": "score", "criteria": ["lo", "hi"]}) == [0.1, 0.9]
    q = {"type": "choice", "criteria": {"b": "x", "a": "y"}}
    assert answer_probs({"type": "choice", "probabilities": {"a": 0.3, "b": 0.7}}, q) == [0.7, 0.3]
    assert answer_probs({"type": "choice", "probabilities": {"a": 0.3, "b": 0.7}},
                        {"type": "choice", "criteria": ["a", "b"]}) == [0.3, 0.7]


def test_compute_metrics_groups():
    m = compute_metrics(_rows())
    assert set(m["by_task"]) == {"t1", "t2"} and set(m["by_modality"]) == {"image", "text"}
    assert m["by_modality"]["image"]["accuracy"] == 1.0
    assert "image/choice" in m["by_modality_qtype"]
    assert len(m["overall"]["reliability"]) == 10
    assert "| t1 |" in markdown_table(m, "x")
