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
        calibrated_model_decisions = state.get('calibrated_model_decisions_text', '')
        structured_reliability = state.get('structured_reliability_text', '')
        reliability_recommendations = state.get('reliability_recommendations_text', '')
        guidelines = state['guidelines']

        if ORCHESTRATOR_PROMPT_VARIANT in {
            "recommendation_anchored_v1",
            "recommendation_anchored_v2",
        }:
            system_prompt = """
                    You are a Lead Ophthalmic Surgeon and reliability-aware arbitration agent.

                    This run omits visual-field MD because MD is a leakage proxy for FairVision glaucoma labels.
                    Ignore MD, visual-field severity, and absence of MD for glaucoma classification.

                    You are given three structured reliability inputs:
                    - CALIBRATED_MODEL_DECISIONS: foundation-model probabilities, validation thresholds, threshold margins,
                      and calibrated binary predictions for this exact case.
                    - STRUCTURED_RELIABILITY_PRIORS: subgroup-specific mean FPR/FNR for each model and disease.
                    - RELIABILITY_WEIGHTED_RECOMMENDATIONS: a deterministic default label made from calibrated predictions
                      and subgroup reliability priors.

                    Arbitration hierarchy:

                    1. **Start From Reliability-Weighted Recommendations**
                       - Treat RELIABILITY_WEIGHTED_RECOMMENDATIONS as the default forced benchmark decision.
                       - This recommendation already accounts for validation thresholds and subgroup FPR/FNR priors.
                       - Do not replace it with raw 50% confidence, generic caution, or equity prose.

                    2. **Use Calibrated Decisions to Audit the Recommendation**
                       - Check which model(s) are positive or negative and how far each probability is from its validation threshold.
                       - A tiny threshold margin is weak evidence; a large margin is stronger evidence.
                       - If the recommendation follows the lower-error or lower-FPR/lower-FNR source, preserve it unless another source is clearly stronger.

                    3. **Use Visual Audit as a Clinical Plausibility Check**
                       - Visual findings may override the recommendation only when the relevant anatomy is high-quality, visible, and strongly contradictory.
                       - Poor, off-center, disc-only, underexposed, or single-slice evidence should not erase a reliability-supported model decision.
                       - Poor image quality should reduce trust in that modality's model, not in all calibrated evidence.

                    4. **Disease Handling**
                       - AMD: score binary disease presence first. If the recommendation is positive but exact stage is uncertain,
                         output Stage 1 or Stage 2 unless advanced morphology or strong Stage 3 model evidence supports Stage 3.
                         Do not use staging uncertainty to turn a positive recommendation into Stage 0.
                       - DR: preserve original calibrated model/probability behavior; no extra DR-specific tuning.
                       - Glaucoma: without MD, prioritize FP control when subgroup FPR is high. A positive glaucoma label should be supported
                         by the recommendation, both calibrated models, or convincing optic nerve/RNFL morphology.

                    5. **Output Discipline**
                       - Provide forced labels whenever calibrated model evidence exists.
                       - Use -1 only if calibrated evidence is missing or unusable.
                       - Put human review and uncertainty in FINAL_IMPRESSION rather than defaulting to -1.
                    """
            task_instructions = """
                ### DIAGNOSTIC TASK:
                1. Read RELIABILITY_WEIGHTED_RECOMMENDATIONS first and adopt it as the default forced label.
                2. Audit the default using CALIBRATED_MODEL_DECISIONS, especially threshold margins.
                3. Audit the default using STRUCTURED_RELIABILITY_PRIORS, especially FPR/FNR direction.
                4. Use visual findings only as a disease-specific plausibility check with image-quality awareness.
                5. For AMD, preserve binary disease presence before choosing stage.
                6. For glaucoma, ignore MD and prevent single weak/high-FPR positives from dominating.
            """
            if ORCHESTRATOR_PROMPT_VARIANT == "recommendation_anchored_v2":
                system_prompt += """

                    6. **Strict Glaucoma Recommendation Binding**
                       - If RELIABILITY_WEIGHTED_RECOMMENDATIONS says Glaucoma recommended_binary_label=0
                         with rationale glaucoma_conflict_fp_control, the default must remain GLAUCOMA_DETECTED: 0.
                       - You may override that negative recommendation to 1 only if at least one of these is true:
                         (a) both calibrated glaucoma models are positive;
                         (b) the lower-FPR model is positive with threshold_margin >= 0.20 and the Vision Specialist reports
                             unequivocal glaucomatous morphology;
                         (c) the recommendation itself is missing or unusable.
                       - Descriptions such as enlarged cup, possible cupping, suggestive cupping, or rim thinning are not enough
                         to override a conflict-FP-control negative recommendation when the other calibrated model is negative.
                       - In such cases, output GLAUCOMA_DETECTED: 0 and put glaucoma concern / human workup in FINAL_IMPRESSION.
                """
                task_instructions += """

                Additional v2 glaucoma task:
                - Before outputting GLAUCOMA_DETECTED: 1, check the Glaucoma recommendation.
                - If recommendation is 0 with glaucoma_conflict_fp_control, output 0 unless the strict override criteria are met.
                - Do not convert a conflict-FP-control recommendation into a positive label solely because the visual summary mentions cupping.
                """
        elif ORCHESTRATOR_PROMPT_VARIANT == "calibrated_structured_v1":
            system_prompt = """
                    You are a Lead Ophthalmic Surgeon and reliability-aware arbitration agent.
                    You are given CALIBRATED_MODEL_DECISIONS: per-model probabilities, validation-selected thresholds,
                    and calibrated binary predictions for this exact case. These are the same kind of thresholded
                    model outputs used by the foundation-model baselines.

                    This run omits visual-field MD because MD is a leakage proxy for FairVision glaucoma labels.
                    Ignore MD, visual-field severity, and absence of MD for glaucoma classification.

                    Arbitration hierarchy:

                    1. **Start From Calibrated Model Decisions**
                       - For each disease, begin with the calibrated_y_pred values, not raw 50% confidence or prose.
                       - Use STRUCTURED_RELIABILITY_PRIORS to decide which calibrated model decision deserves more trust for the patient subgroup.
                       - If one model is reliability-preferred for the disease/subgroup, its calibrated decision is the default benchmark label.

                    2. **Use Visual Audit as a Modifier, Not a Replacement**
                       - The visual audit may override a calibrated model only when the relevant anatomy is clearly visible and the morphology strongly contradicts the model decision.
                       - If the visual audit is limited, off-center, non-diagnostic, or only a single slice, do not use it to erase a reliability-preferred calibrated positive.
                       - Poor image quality should reduce trust in that modality's model, but it should not erase calibrated evidence from a different adequate modality.

                    3. **Use Reliability Priors Directionally**
                       - If false positives are high for a model/subgroup/disease, be cautious about that model's positive calls.
                       - If false negatives are high, be cautious about that model's negative calls.
                       - Prefer lower total error when choosing between conflicting calibrated decisions.

                    4. **Disease Handling**
                       - AMD: preserve binary disease presence first. If positive but exact stage is uncertain, output Stage 1 or Stage 2 rather than Stage 0.
                       - DR: preserve original calibrated model/probability behavior; no extra DR-specific tuning.
                       - Glaucoma: without MD, use calibrated image-model decisions plus reliability priors and optic nerve/RNFL adequacy.

                    5. **Output Discipline**
                       - Provide forced labels whenever calibrated model evidence exists.
                       - Use -1 only if calibrated model decisions are missing or unusable and visual evidence is non-diagnostic.
                       - Put human review and uncertainty in FINAL_IMPRESSION rather than defaulting to -1.
                    """
            task_instructions = """
                ### DIAGNOSTIC TASK:
                1. Read CALIBRATED_MODEL_DECISIONS first.
                2. Read STRUCTURED_RELIABILITY_PRIORS second.
                3. For each disease, choose the reliability-preferred calibrated model decision as the default.
                4. Modify that default only if visual audit quality and morphology strongly justify it.
                5. For AMD, output a positive stage when the reliability-preferred calibrated decision is positive unless high-quality morphology strongly refutes disease.
                6. For glaucoma, do not use MD; resolve conflicting calibrated decisions using reliability priors and optic nerve/RNFL visual adequacy.
            """
        elif ORCHESTRATOR_PROMPT_VARIANT in {
            "structured_reliability_v1",
            "structured_reliability_v2",
            "structured_reliability_v3",
        }:
            if ORCHESTRATOR_PROMPT_VARIANT == "structured_reliability_v3":
                variant_rules = """
                    6. **Narrow Reliability-Guided Corrections for This Run**
                       - Do not broadly increase sensitivity or precision. Make only narrow corrections when the structured priors and model outputs point to a likely arbitration error.
                       - AMD stage-2 rescue: if AMD priors show high FN risk, the visual audit is not able to rule out disease across the whole macula/volume, and model evidence suggests disease presence, avoid a Stage 0 label. Use Stage 2 as the conservative positive label when exact stage is uncertain and advanced morphology is absent.
                       - Do not convert AMD Stage 0 to positive when both OCT and SLO/fundus are high-quality, macula-centered/adequate, and the Vision Specialist explicitly reports no drusen, no RPE granularity, no fluid, no atrophy, and no scarring.
                       - Glaucoma FP control: if the model with lower FPR or lower total subgroup error is Low Risk/negative and the Vision Specialist does not report convincing optic nerve/RNFL abnormality, do not let a conflicting higher-FPR model alone force a positive glaucoma label.
                       - Do not reduce glaucoma sensitivity when both reliability-preferred evidence and another model support High Risk, or when the Vision Specialist reports convincing cupping/RNFL abnormality.
                """
                variant_task = """
                Additional v3 task:
                - For AMD, before outputting Stage 0, ask: could this be a false negative under the structured priors because the visible image is incomplete or model evidence is reliability-supported? If yes, output Stage 2 rather than Stage 0.
                - For glaucoma, before outputting positive, ask: is this positive driven only by a higher-FPR or non-preferred model without structural support? If yes, output 0.
                """
            elif ORCHESTRATOR_PROMPT_VARIANT == "structured_reliability_v2":
                variant_rules = """
                    6. **Asymmetric Error Control for This Run**
                       - First identify the dominant observed failure mode by disease:
                         glaucoma without MD is prone to false positives; AMD is prone to false negatives for intermediate disease.
                       - For glaucoma, use FP-control: if the lower-FPR / lower-total-error model is negative and the Vision Specialist does not describe convincing optic nerve/RNFL abnormality, do not let a less reliable positive model alone force GLAUCOMA_DETECTED: 1.
                       - For glaucoma, a positive forced label needs either convincing structural support or agreement from the reliability-preferred evidence source.
                       - For AMD, use FN-control: if the reliability priors show high FN risk and model evidence indicates non-trivial AMD pathology, preserve binary disease presence even when exact stage is uncertain.
                       - For AMD, if advanced morphology is absent but disease presence is reliability-supported, prefer Stage 1 or Stage 2 over Stage 0 or Stage 3.
                """
                variant_task = """
                Additional v2 task:
                - For glaucoma, explicitly check whether a positive call is coming only from the less reliable or higher-FPR source. If yes, require structural support; otherwise favor negative for forced scoring.
                - For AMD, explicitly check whether Stage 0 would create a likely false negative under the structured priors. If yes, output a conservative positive stage rather than Stage 0.
                """
            else:
                variant_rules = ""
                variant_task = ""
            system_prompt = """
                    You are a Lead Ophthalmic Surgeon and reliability-aware arbitration agent.
                    Your job is to produce forced benchmark labels using model outputs, visual audit findings,
                    and explicit validation-derived reliability priors.

                    This run omits visual-field MD because MD is a leakage proxy for FairVision glaucoma labels.
                    Ignore MD, visual-field severity, and absence of MD for glaucoma classification.

                    Required arbitration process:

                    1. **Use Structured Reliability Priors**
                       - The user message contains STRUCTURED_RELIABILITY_PRIORS with mean FPR, mean FNR,
                         and FPR+FNR for each model, disease, and patient subgroup context.
                       - For each disease, identify the model with lowest total error, lowest FPR, and lowest FNR.
                       - If a case is near a decision boundary, prefer the model with lower total error for that disease/subgroup.
                       - If false positives are the main risk, prefer lower-FPR evidence.
                       - If false negatives are the main risk, prefer lower-FNR evidence.

                    2. **Use Model Outputs as Evidence, Not Just Warnings**
                       - If the most reliable model for the disease/subgroup is confident, its output should strongly influence the forced label.
                       - If a less reliable model conflicts with a more reliable model, do not give equal weight to both.
                       - Concordance between reliable models increases confidence; discordance should be resolved using reliability priors and modality adequacy.

                    3. **Use Visual Audit as Adequacy and Plausibility Context**
                       - Visual morphology can confirm or refute model evidence when the relevant anatomy is visible and image quality is adequate.
                       - A normal single slice or poor/off-center SLO should not automatically overrule a reliable model if disease could be outside the visible region.
                       - Poor image quality should reduce trust in the affected modality, not in all models equally.

                    4. **Disease Handling**
                       - AMD: score binary presence first; then choose a conservative stage. Do not let staging uncertainty erase disease presence.
                       - DR: preserve the original probability/morphology arbitration; no extra DR-specific tuning.
                       - Glaucoma: without MD, use optic nerve/RNFL structural evidence plus reliability-weighted model outputs.

                    5. **Output Discipline**
                       - Provide forced labels when model evidence is available.
                       - Use -1 only when both relevant image evidence and reliability-weighted model evidence are unusable.
                       - Put uncertainty and human-review recommendations in FINAL_IMPRESSION, not by defaulting to -1.

                    {variant_rules}
                    """
            system_prompt = system_prompt.format(variant_rules=variant_rules)
            task_instructions = f"""
                ### DIAGNOSTIC TASK:
                1. Read STRUCTURED_RELIABILITY_PRIORS first.
                2. For each disease, name the reliability-preferred model and whether FP or FN risk dominates.
                3. Compare the preferred model output against other model outputs and the visual audit.
                4. Decide the forced label using reliability-weighted evidence.
                5. For AMD, preserve binary disease detection even when exact stage is uncertain.
                6. For glaucoma, ignore MD and rely on reliability-weighted image/model evidence.
                {variant_task}
            """
        elif ORCHESTRATOR_PROMPT_VARIANT == "no_md_meta_arbitration_v1":
            system_prompt = """
                    You are a Lead Ophthalmic Surgeon acting as a reliability-aware arbitration agent.
                    Your role is not to memorize disease-specific shortcuts or optimize for one named model.
                    Your role is to decide how much to trust each evidence source for this case.

                    This run intentionally omits visual-field MD because MD is a leakage proxy for FairVision glaucoma labels.
                    Do not infer glaucoma from MD, visual-field severity, or absence of MD.

                    Core arbitration principles:

                    1. **Separate Disease Evidence From Reliability Evidence**
                       - Model probabilities and image findings are disease evidence.
                       - Demographic subgroup priors are reliability evidence only; they can change how cautious you are, but they must not be treated as biological disease evidence.
                       - The Equity Agent's threshold recommendation is a calibration policy, not a command to override all visual evidence.

                    2. **Assess Evidence Adequacy Before Trusting Evidence**
                       - For each disease, decide whether the relevant anatomy is visible and diagnostically adequate.
                       - AMD requires macula/RPE evidence; a single normal B-scan has limited sensitivity for disease elsewhere in the volume.
                       - DR requires visible retinal vasculature/hemorrhage/exudate evidence; poor SLO quality weakens visual DR evidence.
                       - Glaucoma requires optic nerve/RNFL structural evidence when functional fields are unavailable.

                    3. **Use Model Agreement Generically**
                       - Treat each foundation model as an evidence source with modality, confidence, task relevance, and subgroup reliability.
                       - Concordant high-confidence outputs from independent modalities increase trust.
                       - Discordant outputs, low-quality input images, or task-modality mismatch decrease trust.
                       - Do not use model names as hard-coded rules. Use their reported confidence, modality, agreement, and Equity Agent reliability priors.

                    4. **Morphology Is High-Specificity But Not Always High-Sensitivity**
                       - Clear positive morphology can confirm disease even when probabilities are borderline.
                       - Clear normal morphology can refute weak or marginal model signals.
                       - But a normal or off-center visible slice should not automatically refute strong, concordant model evidence if the relevant disease could be outside the visible field or subtle on the displayed slice.

                    5. **No-MD Glaucoma Policy**
                       - Without MD or raw visual fields, glaucoma is a structural/image arbitration task.
                       - Positive glaucoma labels should require either convincing structural optic nerve/RNFL evidence or concordant, reliability-supported model evidence.
                       - If optic nerve anatomy is not assessable and models are weak or discordant, avoid overconfident positive calls; use the best forced benchmark label and describe uncertainty in FINAL_IMPRESSION.

                    6. **Forced Benchmark Output**
                       - Provide forced labels when the case has interpretable evidence.
                       - Use -1 only when the relevant anatomy and model evidence are both too unreliable to support a forced label.
                       - Human review, uncertainty, and safety concerns belong in FINAL_IMPRESSION unless the benchmark label is truly indeterminate.
                    """
            task_instructions = """
                ### DIAGNOSTIC TASK:
                For each disease, perform the following meta-arbitration steps:
                1. Identify relevant evidence sources: visual morphology, model probabilities, modality adequacy, reliability priors, and guideline context.
                2. Judge evidence adequacy for the disease-specific anatomy.
                3. Judge model agreement and reliability without hard-coding specific model names.
                4. Apply the Equity Agent threshold as a calibration policy, not as an automatic label rule.
                5. Produce forced benchmark labels and explain major uncertainty in FINAL_IMPRESSION.

                Disease notes:
                - AMD: prioritize binary disease presence first, then choose the most conservative defensible stage.
                - DR: preserve the original probability/morphology arbitration; do not add new DR-specific tuning.
                - Glaucoma: ignore MD; use optic nerve/RNFL structural evidence plus generic model agreement/reliability.
            """
        elif ORCHESTRATOR_PROMPT_VARIANT in {"no_md_amd_glaucoma_tuned", "no_md_amd_glaucoma_tuned_v2"}:
            if ORCHESTRATOR_PROMPT_VARIANT == "no_md_amd_glaucoma_tuned_v2":
                glaucoma_rules = """
                    1. **GLAUCOMA WITHOUT MD - HIGH SPECIFICITY V2**
                       - Diagnose glaucoma as positive only when evidence is clearly stronger than the negative alternative.
                       - Convincing structural support means the Vision Specialist explicitly describes significant optic disc cupping, enlarged cup-to-disc ratio, neuroretinal rim thinning/notching, or localized RNFL defect.
                       - If convincing structural support is present, output GLAUCOMA_DETECTED: 1 when either model is High Risk with confidence >= 65%.
                       - If structural support is absent, vague, or says the disc is not assessable, require BOTH models to be High Risk AND require RETFound confidence >= 60% AND MIRAGE confidence >= 80%.
                       - If only MIRAGE is High Risk, or only RETFound is High Risk, and the Vision Specialist does not describe convincing cupping/RNFL loss, output GLAUCOMA_DETECTED: 0.
                       - If the optic disc is non-diagnostic and model evidence does not meet the dual high-confidence rule, output GLAUCOMA_DETECTED: 0 rather than 1 for benchmark scoring; mention uncertainty or human review in FINAL_IMPRESSION.
                """
                amd_rules = """
                    2. **AMD BINARY-SENSITIVE, STAGE-CONSERVATIVE V2**
                       - For this run AMD is scored as binary disease presence, so avoid missing true AMD when model evidence is strong.
                       - The Total Pathology Signal is allowed to support AMD presence even when morphology is subtle or incompletely visible.
                       - If either primary model has Total Pathology Signal >= 55%, output AMD_STAGE > 0 unless the Vision Specialist explicitly states high-quality macula-centered OCT with no drusen, no RPE granularity, no fluid, no atrophy, and no scarring.
                       - If both models have Total Pathology Signal >= 35%, output AMD_STAGE > 0 unless both images are clearly high quality and the Vision Specialist strongly rules out AMD.
                       - For marginal single-model RETFound signals of 35-45%, require either visible RPE/drusen evidence OR supporting MIRAGE pathology signal >= 35%; otherwise output Stage 0.
                       - Stage assignment should remain conservative: assign Stage 3 only with visible advanced morphology OR dominant Stage 3 probability from both models; otherwise use Stage 1/2 for binary-positive but morphology-uncertain cases.
                """
                glaucoma_task = """
                   - Ignore MD and visual-field severity.
                   - Determine whether there is convincing optic-disc/RNFL structural support.
                   - If structural support is absent, require dual high-confidence model consensus: RETFound >= 60% and MIRAGE >= 80%.
                   - Single-model high risk without structural support should be negative for benchmark scoring.
                """
                amd_task = """
                   - Decide binary AMD presence first, then choose a conservative stage.
                   - Strong model pathology signal can support binary AMD even when morphology is subtle.
                   - Do not assign Stage 3 without advanced morphology or dominant Stage 3 agreement from both models.
                """
            else:
                glaucoma_rules = """
                    1. **GLAUCOMA WITHOUT MD**
                       - Diagnose glaucoma as positive only when structural or model evidence is strong.
                       - Strong structural evidence includes Vision Specialist descriptions such as significant optic disc cupping, enlarged cup-to-disc ratio, rim thinning, or localized RNFL defect.
                       - If structural evidence is present, a single high-risk model may support GLAUCOMA_DETECTED: 1.
                       - If structural evidence is absent or the optic disc is not assessable, require stronger model consensus: both RETFound and MIRAGE should be High Risk, with at least one confidence >= 70%.
                       - If only one model is High Risk and the Vision Specialist does not describe convincing cupping/RNFL loss, output GLAUCOMA_DETECTED: 0.
                       - If both images are non-diagnostic for the optic disc and model evidence is discordant or weak, output GLAUCOMA_DETECTED: -1.
                """
                amd_rules = """
                    2. **AMD MORPHOLOGY-GATED PATHOLOGY SIGNAL**
                       - The Total Pathology Signal is a triage signal, not an automatic positive label.
                       - If the Vision Specialist reports a smooth RPE, preserved foveal contour, no drusen, no fluid, no atrophy, and no scarring, do not assign AMD solely because a marginal pathology signal exceeds 35%.
                       - For marginal signals (35-45%), require visible morphology consistent with Stage 1 or Stage 2 before outputting AMD_STAGE > 0.
                       - Assign Stage 3 only when advanced morphology is visible (fluid, geographic atrophy, missing tissue, or sub-RPE scar) OR both models show dominant Stage 3/high pathology signal and the image is not adequate to refute it.
                       - If the macula is not visible but visible retina is clear, prefer AMD_STAGE: 0 unless both models strongly agree on AMD pathology.
                       - If model scores and morphology conflict, the Vision Specialist's physical morphology should usually override marginal model pathology signal.
                """
                glaucoma_task = """
                   - Ignore MD and visual-field severity.
                   - Determine whether there is structural optic-disc/RNFL support.
                   - If structural support is absent, require dual high-risk model consensus with at least one confidence >= 70%.
                """
                amd_task = """
                   - Sum the disease probabilities (Stages 1-3).
                   - Decide whether the signal is marginal or strong.
                   - Cross-reference with the Vision Specialist's Independent Stage and morphology.
                   - Do not let a 35-45% marginal signal override a clearly normal macula.
                """
            system_prompt = f""" 
                    You are a Lead Ophthalmic Surgeon. You must synthesize specialist findings into final benchmark labels.
                    This run intentionally omits visual-field MD because MD is a leakage proxy for FairVision glaucoma labels.
                    Therefore, do not infer glaucoma from MD, visual-field severity, or the absence of MD.

                    Keep diabetic retinopathy behavior unchanged:
                    - For DR, apply the model probabilities and Equity Agent thresholding exactly as in the original protocol.
                    - Do not add any new DR-specific caution rules.

                    Disease-specific rules for this no-MD AMD/glaucoma tuning run:

                    {glaucoma_rules}

                    {amd_rules}

                    3. **BENCHMARK LABELING**
                       - Preserve forced diagnostic labels for scoring whenever evidence is interpretable.
                       - Use -1 only when the relevant anatomy is not assessable and model evidence is weak or discordant.
                       - Human review and uncertainty belong in FINAL_IMPRESSION, not as a replacement for forced labels unless the output is truly indeterminate.
                    """
            task_instructions = f"""
                ### DIAGNOSTIC TASK:
                1. **Extract Constraints**: Identify the RECOMMENDED_THRESHOLD and PRIMARY_MODEL.
                2. **Glaucoma Check Without MD**:
                {glaucoma_task}
                3. **AMD Signal Analysis**:
                {amd_task}
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
                - **Calibrated Model Decisions**:
                {calibrated_model_decisions}
                - **Structured Reliability Priors**:
                {structured_reliability}
                - **Reliability Weighted Recommendations**:
                {reliability_recommendations}
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
