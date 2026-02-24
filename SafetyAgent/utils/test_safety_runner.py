import json
from agents.safety_agent import SafetyAgent


def main() -> None:
    pipeline_state = {
        "patient_context": {
            "age": 72,
            "risk_factors": ["family_history_glaucoma", "high_iop"],
            "eye": "right",
        },
        "imaging_agents": {
            "oct_summary": {
                "rnfl_thickness_global": 70,
                "flag": "borderline",
            },
            "fundus_quality": {
                "quality_score": 0.82,
                "artifacts": False,
            },
        },
        "functional_agents": {
            "visual_field": {
                "mean_deviation_db": -8.05,
                "reliability": "acceptable",
            }
        },
        "diagnostic_models": {
            "glaucoma_classifier": {
                "ai_score": 0.062,  # 6.2%
                "label": "healthy",
            }
        },
        "final_pipeline_label": "healthy",
    }

    clinician_notes = (
        "Long-standing ocular hypertension, now with progressive visual "
        "field loss over 2 years."
    )

    agent = SafetyAgent()

    print("Sending the following aggregated pipeline JSON to the safety agent:\n")
    print(json.dumps(pipeline_state, indent=2))
    print("\n--- SAFETY AGENT RESPONSE ---\n")

    result = agent.run(pipeline_state, clinician_notes=clinician_notes)
    print(result)


if __name__ == "__main__":
    main()
