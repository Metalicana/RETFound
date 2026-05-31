import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import re


load_dotenv() 
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT = "gpt-5.1"
ORCHESTRATOR_PROMPT_VARIANT = os.environ.get("EQUI_AGENT_ORCHESTRATOR_PROMPT_VARIANT", "default")
        
class Orchestrator:
    def __init__(self, model_client=None):
        
        self.model_client = AzureOpenAI(
          azure_endpoint = endpoint, 
          api_key = api_key,
          api_version="2024-12-01-preview"
          )

    def analyze(self, state, retfound_scores, mirage_scores):
        
        narrative = state['clinical_narrative']
        vision_summary = state['vision_opinion']['summary']
        functional_summary = state['functional_opinion']['summary']
        equity_output = state['equity_opinion']
        guidelines = state['guidelines']

        if ORCHESTRATOR_PROMPT_VARIANT == "no_md_amd_glaucoma_tuned":
            system_prompt = """ 
                    You are a Lead Ophthalmic Surgeon. You must synthesize specialist findings into final benchmark labels.
                    This run intentionally omits visual-field MD because MD is a leakage proxy for FairVision glaucoma labels.
                    Therefore, do not infer glaucoma from MD, visual-field severity, or the absence of MD.

                    Keep diabetic retinopathy behavior unchanged:
                    - For DR, apply the model probabilities and Equity Agent thresholding exactly as in the original protocol.
                    - Do not add any new DR-specific caution rules.

                    Disease-specific rules for this no-MD AMD/glaucoma tuning run:

                    1. **GLAUCOMA WITHOUT MD**
                       - Diagnose glaucoma as positive only when structural or model evidence is strong.
                       - Strong structural evidence includes Vision Specialist descriptions such as significant optic disc cupping, enlarged cup-to-disc ratio, rim thinning, or localized RNFL defect.
                       - If structural evidence is present, a single high-risk model may support GLAUCOMA_DETECTED: 1.
                       - If structural evidence is absent or the optic disc is not assessable, require stronger model consensus: both RETFound and MIRAGE should be High Risk, with at least one confidence >= 70%.
                       - If only one model is High Risk and the Vision Specialist does not describe convincing cupping/RNFL loss, output GLAUCOMA_DETECTED: 0.
                       - If both images are non-diagnostic for the optic disc and model evidence is discordant or weak, output GLAUCOMA_DETECTED: -1.

                    2. **AMD MORPHOLOGY-GATED PATHOLOGY SIGNAL**
                       - The Total Pathology Signal is a triage signal, not an automatic positive label.
                       - If the Vision Specialist reports a smooth RPE, preserved foveal contour, no drusen, no fluid, no atrophy, and no scarring, do not assign AMD solely because a marginal pathology signal exceeds 35%.
                       - For marginal signals (35-45%), require visible morphology consistent with Stage 1 or Stage 2 before outputting AMD_STAGE > 0.
                       - Assign Stage 3 only when advanced morphology is visible (fluid, geographic atrophy, missing tissue, or sub-RPE scar) OR both models show dominant Stage 3/high pathology signal and the image is not adequate to refute it.
                       - If the macula is not visible but visible retina is clear, prefer AMD_STAGE: 0 unless both models strongly agree on AMD pathology.
                       - If model scores and morphology conflict, the Vision Specialist's physical morphology should usually override marginal model pathology signal.

                    3. **BENCHMARK LABELING**
                       - Preserve forced diagnostic labels for scoring whenever evidence is interpretable.
                       - Use -1 only when the relevant anatomy is not assessable and model evidence is weak or discordant.
                       - Human review and uncertainty belong in FINAL_IMPRESSION, not as a replacement for forced labels unless the output is truly indeterminate.
                    """
            task_instructions = """
                ### DIAGNOSTIC TASK:
                1. **Extract Constraints**: Identify the RECOMMENDED_THRESHOLD and PRIMARY_MODEL.
                2. **Glaucoma Check Without MD**:
                   - Ignore MD and visual-field severity.
                   - Determine whether there is structural optic-disc/RNFL support.
                   - If structural support is absent, require dual high-risk model consensus with at least one confidence >= 70%.
                3. **AMD Signal Analysis**:
                   - Sum the disease probabilities (Stages 1-3).
                   - Decide whether the signal is marginal or strong.
                   - Cross-reference with the Vision Specialist's Independent Stage and morphology.
                   - Do not let a 35-45% marginal signal override a clearly normal macula.
                4. **DR Check**:
                   - Use the original DR probability/threshold behavior; no new DR-specific rules are introduced in this prompt.
                5. **Final Staging**:
                   - Choose the most defensible forced labels under the disease-specific rules above.
            """
        else:
            system_prompt = """ 
                    You are a Lead Ophthalmic Surgeon. You must synthesize specialist findings into a final diagnosis using the following strict hierarchy:
                    
                    1. **THE GLAUCOMA "FUNCTIONAL BRAKE"** (Mandatory)
                       - If the Functional Agent reports a Normal field (MD > -2.0 dB), you must output 0 for Glaucoma.
                    
                    2. **THE AMD "PATHOLOGY SIGNAL" RULE**
                       - Do not just look at the "Primary Prediction" stage. Analyze the "[!] TOTAL PATHOLOGY SIGNAL" (the sum of Stages 1, 2, and 3).
                       - If the Total Pathology Signal exceeds the RECOMMENDED_THRESHOLD (e.g., 35%) from the Equity Agent, you must consider the case                           POSITIVE for AMD.
                       - To determine the stage: Look at the distribution mass. If the mass is concentrated in Stage 2, and the Vision Specialist confirms                       "lumpy" or "textured" RPE, you must assign Stage 2.
                    
                    3. **THE AMD "CONSENSUS RULE"**
                       - To prevent "Stage 0" calls for subtle Stage 2 cases:
                       - If BOTH RetFound and MIRAGE show a Total Pathology Signal > 60%, do NOT output Stage 0, even if the Vision Specialist reports a                         clear image.
                       - In this case, defer to the specific stage (1, 2, or 3) that has the highest relative probability among the disease stages.
                    
                    4. **DYNAMIC THRESHOLDING & WEIGHTING**
                       - Apply the RECOMMENDED_THRESHOLD from the Equity Audit to all AMD and DR probabilities.
                       - Prioritize the PRIMARY_MODEL identified by the Equity Agent during conflicts.
                    
                    5. THE "PROXY" EVALUATION AND BENCHMARK LABELING:
                    - If the Vision Specialist says "Macula not in view," check the AI Models.       
                    - If RETFound (OCT) and MIRAGE (SLO) both agree on a stage (e.g., both say Stage 2), accept the AI's diagnosis. 
                    - For benchmark scoring, always provide forced diagnostic labels whenever any model evidence is available.
                    - Human escalation/safety concern belongs in FINAL_IMPRESSION, not as an abstention label.
                    - Only output -1 if the Vision Specialist says "Image Unreadable" AND both AI models have confidence < 40%. """
            task_instructions = """
                ### DIAGNOSTIC TASK:
                1. **Extract Constraints**: Identify the RECOMMENDED_THRESHOLD and PRIMARY_MODEL.
                2. **Glaucoma Check**: Apply the "Functional Brake" (MD > -2.0 dB check).
                3. **AMD Signal Analysis**: 
                   - Sum the disease probabilities (Stages 1-3). 
                   - Does this "Total Pathology Signal" exceed the threshold?
                   - If yes, cross-reference with the Vision Specialist's "Independent Stage" and morphological descriptions.
                4. **Final Staging**: Use the Consensus Rule to ensure intermediate cases aren't downgraded to Stage 0.
            """
    
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"""
                ### MULTI-AGENT CASE INPUTS
                - **Clinical Narrative**: {narrative}
                - **RetFound (OCT) Distribution**: {retfound_scores}
                - **MIRAGE (SLO) Distribution**: {mirage_scores}
                - **Vision Specialist Findings**: {vision_summary}
                - **Functional Specialist Findings**: {functional_summary}
                - **Equity Agent Audit**: {equity_output}
                - **Clinical Guidelines**: {guidelines}

                {task_instructions}
                
                ### OUTPUT FORMAT
                [LABELS]
                AMD_STAGE: [-1, 0, 1, 2, or 3]
                DR_DETECTED: [-1, 0 or 1]
                GLAUCOMA_DETECTED: [-1, 0 or 1]
                [/LABELS]
                
                FINAL_IMPRESSION: [One-sentence clinical summary of the final decision logic. If human review is recommended, state it here while preserving the forced labels above.]
        
                """
            }
        ]

        response = self.model_client.chat.completions.create(
            model=DEPLOYMENT,
            messages=messages,
            temperature=0.3
        )

        raw_response = response.choices[0].message.content
        
        query_match = re.search(r"\[QUERY\](.*?)\[/QUERY\]", raw_response, re.DOTALL)
        if query_match:
            pubmed_query = query_match.group(1).strip()
            final_decision = re.sub(r"\[QUERY\].*?\[/QUERY\]", "", raw_response, flags=re.DOTALL).strip()
        else:
            pubmed_query = "Ophthalmology treatment guidelines 2026"
            final_decision = raw_response
            
        label_match = re.search(r"\[LABELS\](.*?)\[/LABELS\]", raw_response, re.DOTALL)
        if label_match:
            output_labels = label_match.group(1).strip()
        else:
            output_labels = "AMD_STAGE: 0, DR_DETECTED: 0, GLAUCOMA_DETECTED: 0"
        
        return {
            "agent": "Ophthalmic_Master",
            "decision": final_decision,
            "labels": output_labels,
            "pubmed_query": pubmed_query
        }
