from laya_vision.data.text_eval import convert_ag_news, convert_boolq


def test_ag_news_one_hot_in_criteria_order():
    recs = list(convert_ag_news([{"text": "Stocks fell", "label": 2}]))
    assert recs[0]["target"] == [0.0, 0.0, 1.0, 0.0]
    assert recs[0]["image"] is None and recs[0]["split"] == "test"


def test_boolq_noul_target_is_false_true():
    rows = [{"question": "is the sky blue", "answer": True, "passage": "The sky is blue."},
            {"question": "is grass red?", "answer": False, "passage": "Grass is green."}]
    recs = list(convert_boolq(rows))
    assert [r["target"] for r in recs] == [[0.0, 1.0], [1.0, 0.0]]
    assert recs[0]["question"]["instructions"].endswith("Is the sky blue?")
