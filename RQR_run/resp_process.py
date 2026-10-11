SCHEMA_KEYS = ("entities", "unary_facts", "binary_facts", "negated_facts")


def clean_facts(result):
    return {key: result[key] for key in SCHEMA_KEYS}


def normalize_judgement(result):
    return {
        "final_answer": str(result["final_answer"]).upper(),
        "confidence": float(result["confidence"]),
        "reasons": [str(r) for r in result["reasons"]],
    }
