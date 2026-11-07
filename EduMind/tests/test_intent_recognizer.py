import pytest

from core.intent_recognizer import IntentCategory, IntentRecognizer


def make_recognizer(threshold=0.5):
    recognizer = IntentRecognizer.__new__(IntentRecognizer)
    recognizer._embedding_enabled = True
    recognizer.threshold = threshold
    return recognizer


def vote(recognizer, llm, emb, pat):
    return recognizer._vote(
        {"intent": llm[0], "confidence": llm[1]},
        {"intent": emb[0], "confidence": emb[1]},
        {"intent": pat[0], "confidence": pat[1]},
    )


def test_pattern_cannot_refine_a_different_intent_group():
    # 学习计划（request 组）胜出时，作业辅导（query 组）的 pattern 命中不应改写结果。
    intent, confidence, sources = vote(
        make_recognizer(),
        (IntentCategory.STUDY_PLAN, 0.9),
        (IntentCategory.OTHER, 0.0),
        (IntentCategory.HOMEWORK_HELP, 0.5),
    )

    assert intent is IntentCategory.STUDY_PLAN
    assert confidence == pytest.approx(0.63)
    assert "refined_by_pattern" not in sources


def test_coarse_and_fine_scores_choose_the_same_group_together():
    # 细意图所在分组票数更高时，即使 LLM 给的是另一个粗意图，也应落在该分组内的细意图上。
    intent, confidence, sources = vote(
        make_recognizer(threshold=0.2),
        (IntentCategory.STUDY_PLAN, 0.3),          # weighted score 0.21（request 组）
        (IntentCategory.KNOWLEDGE_QUERY, 0.9),     # weighted score 0.18（query 组）
        (IntentCategory.QUERY, 0.4),               # weighted score 0.04（query 组）
    )

    assert intent is IntentCategory.KNOWLEDGE_QUERY
    assert confidence == pytest.approx(0.22)
    assert "refined_by_pattern" not in sources


def test_pattern_can_still_refine_within_the_winning_group():
    # 学生说“这个不会”时，LLM 容易只给出宽泛的 query，pattern 应把它细化为作业辅导。
    intent, confidence, sources = vote(
        make_recognizer(),
        (IntentCategory.QUERY, 0.8),
        (IntentCategory.OTHER, 0.0),
        (IntentCategory.HOMEWORK_HELP, 0.5),
    )

    assert intent is IntentCategory.HOMEWORK_HELP
    assert confidence == pytest.approx(0.61)
    assert sources["refined_by_pattern"] == pytest.approx(0.5)
