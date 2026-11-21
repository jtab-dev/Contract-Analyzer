import json
from contract_engine import analyze_contract_deterministic
from samples import SAMPLE_DOCUMENTS, SAMPLE_FINANCIAL_BACKGROUND, SAMPLE_MEDICAL_HISTORY


def run_tests() -> None:
    for doc in SAMPLE_DOCUMENTS:
        result = analyze_contract_deterministic(
            extracted_text=doc["rawText"],
            medical_text=SAMPLE_MEDICAL_HISTORY,
            financial_text=SAMPLE_FINANCIAL_BACKGROUND,
        )
        h = result["healthAnalysis"]
        ml = h["mlSentiment"]
        assert ml["totalClausesAnalyzed"] > 0
        for cls in ml["clauseClassifications"]:
            prob_sum = cls["positiveProb"] + cls["neutralProb"] + cls["negativeProb"]
            assert abs(prob_sum - 1.0) <= 0.01, f"Probabilities did not sum to 1: {prob_sum}"

        print(
            json.dumps(
                {
                    "id": doc["id"],
                    "category": result["category"],
                    "acaCompliant": h["acaCompliant"],
                    "affordabilityVerdict": h["affordability"]["verdict"],
                    "eligibilityVerdict": h["eligibility"]["verdict"],
                    "overallSentiment": ml["overallSentiment"],
                    "counts": {
                        "pos": ml["positiveCount"],
                        "neu": ml["neutralCount"],
                        "neg": ml["negativeCount"],
                    },
                    "topUnfavorable": [
                        f"{u['condition']} ({u['count']})"
                        for u in ml["mostCommonUnfavorableConditions"]
                    ],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    run_tests()
    print("\nALL PYTHON ML & HEALTH POLICY TESTS PASSED.")
