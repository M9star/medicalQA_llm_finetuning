from medicalqa_finetuning.data import format_medmcqa_alpaca, format_pubmedqa_alpaca
from medicalqa_finetuning.evaluate import extract_medmcqa_answer, extract_pubmedqa_answer


def test_format_medmcqa_alpaca_contains_answer_and_metadata():
    example = {
        "question": "Which vitamin deficiency causes scurvy?",
        "opa": "Vitamin A",
        "opb": "Vitamin B12",
        "opc": "Vitamin C",
        "opd": "Vitamin D",
        "cop": 2,
        "exp": "Scurvy is caused by vitamin C deficiency.",
        "subject_name": "Biochemistry",
        "topic_name": "Vitamins",
    }

    record = format_medmcqa_alpaca(example)

    assert "The correct answer is C)" in record["output"]
    assert "### Instruction:" in record["text"]
    assert record["subject"] == "Biochemistry"


def test_format_pubmedqa_alpaca_uses_context_and_decision():
    example = {
        "question": "Does treatment improve survival?",
        "context": {"contexts": ["A study reported improved survival."]},
        "final_decision": "yes",
        "long_answer": "Treatment was associated with improved outcomes.",
    }

    record = format_pubmedqa_alpaca(example)

    assert "A study reported improved survival." in record["input"]
    assert record["output"].startswith("Yes.")


def test_answer_extractors():
    assert extract_medmcqa_answer("The correct answer is B) Influenza.") == "B"
    assert extract_pubmedqa_answer("Maybe. Evidence is mixed.") == "maybe"
