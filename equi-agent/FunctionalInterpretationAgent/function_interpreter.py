import os
import math

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False

try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None


load_dotenv() 
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT = "gpt-5.1"


GDP_TD_COLUMNS = [
    "td1",
    "td2",
    "td3",
    "td4",
    "td5",
    "td6",
    "td7",
    "td8",
    "td9",
    "td10",
    "td11",
    "td12",
    "td13",
    "td14",
    "td15",
    "td16",
    "td17",
    "td18",
    "td19",
    "td20",
    "td21",
    "td22",
    "td23",
    "td24",
    "td26",
    "td27",
    "td28",
    "td29",
    "td30",
    "td31",
    "td32",
    "td33",
    "td35",
    "td36",
    "td37",
    "td38",
    "td39",
    "td40",
    "td41",
    "td42",
    "td43",
    "td44",
    "td45",
    "td46",
    "td47",
    "td48",
    "td49",
    "td50",
    "td51",
    "td52",
    "td53",
    "td54",
]


def _coerce_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def extract_td_values(metadata):
    """Return finite total-deviation values from GDP-style visual-field metadata."""
    td_source = (
        metadata.get("td_values")
        or metadata.get("visual_field_td_values")
        or metadata.get("total_deviation_values")
    )
    values = []
    if isinstance(td_source, dict):
        iterable = [td_source.get(col) for col in GDP_TD_COLUMNS if col in td_source]
    elif isinstance(td_source, (list, tuple)):
        iterable = td_source
    else:
        iterable = [metadata.get(col) for col in GDP_TD_COLUMNS if col in metadata]

    for raw in iterable:
        number = _coerce_float(raw)
        if number is not None:
            values.append(number)
    return values


def compute_md_from_td_values(td_values):
    """Compute an MD-like score from raw total-deviation points.

    Harvard-GDP stores 52 non-blind-spot TD values. The unweighted mean tracks
    the provided MD column closely and keeps the raw-VF pipeline independent
    from precomputed MD at prompt time.
    """
    if not td_values:
        return None
    return sum(td_values) / len(td_values)


def vf_severity(md_value):
    if md_value is None:
        return "Not Available"
    if md_value > -2:
        return "Normal"
    if md_value > -6:
        return "Early"
    if md_value > -12:
        return "Moderate"
    return "Advanced"


def summarize_td_values(td_values):
    if not td_values:
        return {
            "td_point_count": 0,
            "td_min": None,
            "td_max": None,
            "td_mean": None,
            "td_depressed_at_least_2db_count": 0,
            "td_depressed_at_least_5db_count": 0,
            "td_depressed_at_least_10db_count": 0,
        }
    mean_value = sum(td_values) / len(td_values)
    return {
        "td_point_count": len(td_values),
        "td_min": min(td_values),
        "td_max": max(td_values),
        "td_mean": mean_value,
        "td_depressed_at_least_2db_count": sum(1 for value in td_values if value <= -2),
        "td_depressed_at_least_5db_count": sum(1 for value in td_values if value <= -5),
        "td_depressed_at_least_10db_count": sum(1 for value in td_values if value <= -10),
    }
        
class FunctionalSpecialist:
    def __init__(self, model_client=None):
        if AzureOpenAI is None:
            raise ImportError("openai package is required for FunctionalSpecialist LLM mode.")
        
        self.model_client = AzureOpenAI(
          azure_endpoint = endpoint, 
          api_key = api_key,
          api_version="2024-12-01-preview"
          )

    def analyze(self, state):
        
        metadata = state['metadata']
        
        td_values = extract_td_values(metadata)
        computed_md = compute_md_from_td_values(td_values)
        md_value = computed_md if computed_md is not None else _coerce_float(metadata.get('md'))
        md_score = f"{md_value:.2f} dB" if md_value is not None else "Not Available"
        severity = vf_severity(md_value)
        td_summary = summarize_td_values(td_values)
        td_vector_text = "Not Available"
        if td_values:
            td_vector_text = ", ".join(f"{value:.1f}" for value in td_values)
        
        narrative = state['clinical_narrative']
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Specialist at interpreting retinal visual field tests. "
                    "You will be given perimetry data, including raw total-deviation visual-field points when available. "
                    "When raw total-deviation points are provided, use the supplied computed MD-like score derived from those "
                    "points rather than assuming an externally supplied MD value. You translate numerical deficits into clinical severity "
                    "(Early, Moderate, or Advanced loss).\n\n"
                    "CRITICAL: You must conclude your report with a section titled '[EXECUTIVE SUMMARY]' "
                    "containing 3-5 bullet points for the Lead Ophthalmic Surgeon."
                )
            },
            {    
              #visual field data, harvard gdp 54 sized vector
                "role": "user",
                "content": f"""
                ### INPUT DATA
                - **Computed Mean Deviation from Raw TD Points:** {md_score}
                - **Computed Severity:** {severity}
                - **Raw Total-Deviation Point Count:** {td_summary['td_point_count']}
                - **Raw Total-Deviation Vector (dB):** {td_vector_text}
                - **TD Summary:** {td_summary}
                - **Patient Narrative:** {narrative}

                ### TASK
                1. Interpret the computed MD severity. (Normal: > -2dB, Early: -2 to -6dB, Moderate: -6 to -12dB, Advanced: < -12dB).
                2. Explain how this functional loss aligns with the symptoms described in the narrative.
                3. Provide a 'Functional Status Report'.
                4. Conclude with the [EXECUTIVE SUMMARY].
                """
            }
        ]

        response = self.model_client.chat.completions.create(
            model=DEPLOYMENT,
            messages=messages,
            temperature=0.3 # Lower temperature for consistent clinical interpretation
        )
        
        full_content = response.choices[0].message.content
        
        if "[EXECUTIVE SUMMARY]" in full_content:
            summary = full_content.split("[EXECUTIVE SUMMARY]")[-1].strip()
        else:
            summary = full_content # Fallback

        return {
            "agent": "Functional Specialist",
            "full_report": full_content,
            "summary": summary,
            "computed_md_from_td": computed_md,
            "md_used": md_value,
            "severity": severity,
            "td_summary": td_summary,
        }
        
        return 
