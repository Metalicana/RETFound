import sys
import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from openai import AzureOpenAI
from dotenv import load_dotenv
import base64
import io
from VisionAgent.linear_probing_oct3 import get_model_oct

MIRAGE_DIR = os.path.abspath("./VisionAgent/MIRAGE")
sys.path.append(MIRAGE_DIR)

from linear_probing_slo import get_model_slo

load_dotenv() 
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

class VisionSpecialist:
    def __init__(self, path_oct, path_slo, device=None):
        
        ### OCT MODEL ###
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model_oct = get_model_oct()

        if not os.path.exists(path_oct):
            raise FileNotFoundError(f"Weights not found at {path_oct}")
            
        self.model_oct.load_state_dict(torch.load(path_oct, map_location=self.device, weights_only=True))
        self.model_oct.to(self.device)
        self.model_oct.eval()
#        print("OCT Specialist weights loaded successfully.")
        
        
        ### SLO MODEL ###
        original_dir = os.getcwd()
        os.chdir(MIRAGE_DIR)
        self.model_slo = get_model_slo().to(self.device)
        os.chdir(original_dir)
        
        self.model_slo.load_state_dict(torch.load(path_slo, map_location=self.device, weights_only=True))
        self.model_slo.eval()
        
        
        self.model_client = AzureOpenAI(
          azure_endpoint = endpoint, 
          api_key = api_key,
          api_version="2024-12-01-preview"
          )
        self.deployment_name = "gpt-5.1"
        

    def _preprocess_oct(self, oct_img):
    
        image = Image.fromarray(oct_img).convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
        img_tensor = transform(image).unsqueeze(0).to(self.device) 
        
        return img_tensor
        
    
    def _preprocess_slo(self, fundus_img):
        image = Image.fromarray(fundus_img).convert('L')   
        
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
         
        input_tensor = transform(image).unsqueeze(0).to(self.device) 
        
        return input_tensor
        
        
    def get_features_oct(self, oct_img):
        """
        Runs the OCT volume through the specialists and returns a diagnostic report.
        """
        input_tensor = self._preprocess_oct(oct_img)
        
        with torch.no_grad():
            output = self.model_oct(input_tensor)
            
            amd_probs = torch.sigmoid(output['amd']).cpu().numpy()[0]
            
            predicted_stage = 0
            if amd_probs[2] > 0.5: predicted_stage = 3
            elif amd_probs[1] > 0.5: predicted_stage = 2
            elif amd_probs[0] > 0.5: predicted_stage = 1
    
            dr_prob = torch.sigmoid(output['dr']).item()
            glaucoma_prob = torch.sigmoid(output['glaucoma']).item()
            
        return {
            'AMD': {
                'Predicted_Stage': predicted_stage,
                'Early': round(float(amd_probs[0]), 4),
                'Intermediate': round(float(amd_probs[1]), 4),
                'Advanced': round(float(amd_probs[2]), 4)
            },
            'DR': {
                'Probability': round(dr_prob, 4),
                'Detected': dr_prob > 0.5
            },
            'Glaucoma': {
                'Probability': round(glaucoma_prob, 4),
                'Detected': glaucoma_prob > 0.5
            }
        }
        
    
    def get_features_slo(self, fundus_image):
        """
        Runs the SLO through the specialists and returns a diagnostic report.
        """
        input_tensor = self._preprocess_slo(fundus_image)
        
        with torch.no_grad():
          outputs = self.model_slo(input_tensor)
          
          amd_probs = torch.sigmoid(outputs['amd']).cpu().numpy()[0]
          dr_prob = torch.sigmoid(outputs['dr']).cpu().numpy()[0][0]
          glaucoma_prob = torch.sigmoid(outputs['glaucoma']).cpu().numpy()[0][0]
          
          predicted_stage = 0
          if amd_probs[2] > 0.5: predicted_stage = 3
          elif amd_probs[1] > 0.5: predicted_stage = 2
          elif amd_probs[0] > 0.5: predicted_stage = 1
        
            
        return {
            'AMD': {
                'Predicted_Stage': predicted_stage,
                'Early': round(float(amd_probs[0]), 4), #prob AT LEAST early
                'Intermediate': round(float(amd_probs[1]), 4), #prob AT LEAST Intermediate
                'Advanced': round(float(amd_probs[2]), 4) #prob AT LEAST Advanced
            },
            'DR': {
                'Probability': round(dr_prob, 4),
                'Detected': dr_prob > 0.5
            },
            'Glaucoma': {
                'Probability': round(glaucoma_prob, 4),
                'Detected': glaucoma_prob > 0.5
            }
        }
    def _prepare_image(self, numpy_img):
        """Converts numpy array from AgentState to Base64 for Azure."""
        pil_img = Image.fromarray(numpy_img.astype('uint8'))
        buff = io.BytesIO()
        pil_img.save(buff, format="JPEG")
        return base64.b64encode(buff.getvalue()).decode('utf-8')   
          
    def analyze(self, oct_img, fundus_image, state):
      state["oct_diagnosis"] = self.get_features_oct(oct_img)
      state["slo_diagnosis"] = self.get_features_slo(fundus_image)
      
      opinion = self.analyze_visual(state)
    
      return opinion

    def analyze_visual(self, state):
        # 1. Prepare Images
        base64_slo_image = self._prepare_image(state['fundus_img'])
        base64_oct_image = self._prepare_image(state['oct_img'])
        
        # 2. Process OCT and SLO distributions
        for diag_key in ['oct_diagnosis', 'slo_diagnosis']:
            diag = state[diag_key]
            
            # Extract cumulative sigmoid probabilities (0.0 - 1.0)
            p_early = diag['AMD']['Early']
            p_inter = diag['AMD']['Intermediate']
            p_adv   = diag['AMD']['Advanced']
            
            # Calculate Discrete Stage Probabilities (PMF)
            # Using max(0, ...) to guard against noisy model inversions
            s0 = max(0, 1.0 - p_early)
            s1 = max(0, p_early - p_inter)
            s2 = max(0, p_inter - p_adv)
            s3 = p_adv
            
            # Normalize to 100%
            total = s0 + s1 + s2 + s3
            dist = {
                "Stage 0": round((s0/total)*100, 2),
                "Stage 1": round((s1/total)*100, 2),
                "Stage 2": round((s2/total)*100, 2),
                "Stage 3": round((s3/total)*100, 2)
            }
            
            # Store the full distribution in state
            state[diag_key]['AMD']['Distribution'] = dist
            
            # Calculate Total Pathology Signal (Sum of Stages 1-3)
            path_signal = round(dist['Stage 1'] + dist['Stage 2'] + dist['Stage 3'], 2)
            state[diag_key]['AMD']['Pathology_Signal'] = path_signal

            # Identify the "Winning" Stage for the Description
            winning_stage = max(dist, key=dist.get)
            state[diag_key]['AMD']['Description'] = f"Predicted {winning_stage}"
            state[diag_key]['AMD']['Probability'] = dist[winning_stage]

            # 3. Handle DR and Glaucoma (Simple binary)
            for disease in ['DR', 'Glaucoma']:
                prob_pct = diag[disease]['Probability'] * 100
                state[diag_key][disease]['Prob_Pct'] = round(prob_pct, 2)
                
                if disease == 'DR':
                    state[diag_key][disease]['Status'] = "Positive" if prob_pct > 50 else "Negative"
                else: # Glaucoma
                    state[diag_key][disease]['Status'] = "High Risk" if prob_pct > 50 else "Low Risk"

        oct_data = state['oct_diagnosis']
        retfound_scores = (
            f"SOURCE: RETFound (OCT Imaging Specialist)\n"
            f"1. AMD Staging Distribution:\n"
            f"   - Stage 0 (Healthy): {oct_data['AMD']['Distribution']['Stage 0']}%\n"
            f"   - Stage 1 (Early): {oct_data['AMD']['Distribution']['Stage 1']}%\n"
            f"   - Stage 2 (Intermediate): {oct_data['AMD']['Distribution']['Stage 2']}%\n"
            f"   - Stage 3 (Advanced): {oct_data['AMD']['Distribution']['Stage 3']}%\n"
            f"   [!] TOTAL PATHOLOGY SIGNAL (Sum of Stages 1-3): {oct_data['AMD']['Pathology_Signal']}%\n"
            f"   [!] PRIMARY PREDICTION: {oct_data['AMD']['Description']}\n"
            f"2. DR Status: {oct_data['DR']['Status']} (Confidence: {oct_data['DR']['Prob_Pct']}%)\n"
            f"3. Glaucoma Status: {oct_data['Glaucoma']['Status']} (Confidence: {oct_data['Glaucoma']['Prob_Pct']}%)\n"
        )

        slo_data = state['slo_diagnosis']
        mirage_scores = (
            f"SOURCE: MIRAGE (SLO Imaging Specialist)\n"
            f"1. AMD Staging Distribution:\n"
            f"   - Stage 0 (Healthy): {slo_data['AMD']['Distribution']['Stage 0']}%\n"
            f"   - Stage 1 (Early): {slo_data['AMD']['Distribution']['Stage 1']}%\n"
            f"   - Stage 2 (Intermediate): {slo_data['AMD']['Distribution']['Stage 2']}%\n"
            f"   - Stage 3 (Advanced): {slo_data['AMD']['Distribution']['Stage 3']}%\n"
            f"   [!] TOTAL PATHOLOGY SIGNAL (Sum of Stages 1-3): {slo_data['AMD']['Pathology_Signal']}%\n"
            f"   [!] PRIMARY PREDICTION: {slo_data['AMD']['Description']}\n"
            f"2. DR Status: {slo_data['DR']['Status']} (Confidence: {slo_data['DR']['Prob_Pct']}%)\n"
            f"3. Glaucoma Status: {slo_data['Glaucoma']['Status']} (Confidence: {slo_data['Glaucoma']['Prob_Pct']}%)\n"
        )
#            
#        messages = [
#            {
#                "role": "system",
#                "content": (
#                """You are a Senior Ophthalmic Imaging Specialist. Your task is to perform a "Grounded Visual Audit" of retinal images (OCT and SLO/Fundus), cross-referencing your visual findings against AI probability distributions.
#
#### CLINICAL DIAGNOSTIC CRITERIA
#You must use the following morphological benchmarks to determine disease staging. Prioritize your physical visual evidence over the AI's numerical classification.
#
#1. **AMD STAGING (Focus on the RPE-Bruch's membrane interface):**
# - STAGE 0 (Healthy): perfectly smooth RPE; no elevations or granular deposits.
# - STAGE 1 (Early): Small, isolated dots (< 1/2 retinal vein width).
# - STAGE 2 (Intermediate): Multiple distinct bumps OR confluent lumpy/granular texture (1/2 to 1 full retinal vein width).
# - STAGE 3 (Advanced): Fluid, geographic atrophy (missing tissue), or sub-RPE scarring.
#
#2. **DR MARKERS:** Identify microaneurysms, intraretinal hemorrhages, or hard exudates.
#3. **GLAUCOMA MARKERS:** Identify significant optic disc cupping (increased cup-to-disc ratio) or localized RNFL defects.
#
#### AUDIT PROTOCOL
#- **Step 1: Independent Inspection.** Perform an initial independent visual review of the macula (OCT) and optic disc (SLO) for physical lesions.
#- **Step 2: Compare against Pathology Signals.** Read the AI_PROBABILITIES, specifically looking at the full distributions and the `[!] TOTAL PATHOLOGY SIGNAL` (the sum of disease Stages 1-3).
#- **Step 3: Resolve "Subtle Pathology" Tension.** A conflict occurs if you initially see Stage 0 but the AI's Pathology Signal is > 30\%. In this scenario, you must zoom in on the RPE interface and look for subtle "textured granularity" or "pebble-like" bumps (Stage 2) that might be ignored at lower magnification.
#- **Step 4: Identify CRITICAL CONFLICTS.** A conflict is defined if the physical visual markers do not substantiate the AI's primary classification (e.g., AI calls Stage 3 but there is absolutely no visible fluid or atrophy). """
#
#                )
#            },
#            {
#                "role": "user",
#                "content": [
#                    {
#                        "type": "text",
#                        "text": f"""
#                        ### INPUT DATA FOR CASE
#                        1. IMAGE_OCT: [Attached B-scan]
#                        2. IMAGE_SLO: [Attached SLO/Fundus image]
#                        3. AI_PROBABILITIES (Full Distributions):
#                        {retfound_scores}
#                        
#                        {mirage_scores}
#                        
#                        ### YOUR ASSIGNMENT
#                        1. Perform a detailed inspection of the macula foveal contour and the RPE interface on the OCT. Characterize any drusen or RPE                            elevations using the specific clinical benchmarks provided in your instructions. Identify if the macula is centered. If not                               centered (e.g., Disc-centered image), do not abstain. Instead, scan the visible temporal retina for drusen or pigmentary changes.
#                        
#                        Revised Staging: If the macula is not visible, but the visible retina is healthy, report "Stage 0 (Periphery Clear)." Only report                         "Indeterminate" if the image is so blurry or poorly framed that no retina is visible.
#                        
#                        2. In the SLO image, audit the vasculature for hemorrhages/exudates and the optic disc for cupping.
#                        3. Compare your visual staging against the AI distribution. Pay close attention to the `[!] TOTAL PATHOLOGY SIGNAL`. If the signal                        is high, perform a targeted, high-magnification check of the RPE interface for subtle granularity or very small "bumps"                                   (indicating Stage 2).
#                        4. Conclude your report with an Executive Summary detailing your visual evidence and any critical conflicts.
#                        
#                        ### REQUIRED OUTPUT STRUCTURE
#                        You must conclude your report with the following formatted block:
#                        
#                        [EXECUTIVE SUMMARY]
#                        - VISUAL FINDINGS: (e.g., "OCT: smooth RPE, foveal contour preserved. Fundus: normal vasculature, physiological cup. SLO macula                           not visible.")
#                        - INDEPENDENT STAGE: (Identify Stage 0, 1, 2, or 3)
#                        - ALIGNMENT: (Agree / Conflict / Uncertain)
#                        - CONFLICT DETAIL: (If 'Conflict', explain why the physical image morphology does not substantiate the AI's classification or                             pathology signal.)
#                        [/EXECUTIVE SUMMARY]
#                        """
#                    },
#                    { "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_oct_image}"} },
#                    { "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_slo_image}"} }
#                ]
#            }
#        ]
            
        messages = [
            {
                "role": "system",
                "content": (
                """You are a specialized Ophthalmic Imaging Agent. Your sole responsibility is to identify and quantify structural biomarkers in fundus and OCT images.
                **Your Task**:
                Analyze the provided image data for the following:
                1. **Glaucoma**: Cup-to-Disc ratio, RNFL thinning, notch presence
                2. **AMD**: Drusen size (small < 63um, medium 63-125um, large > 125um), pigmentary changes, sub-retinal fluid.
                3. **DR**: Count microaneurysms, hemorrhages, and exudates
                
                Constraint: Do not suggest a final diagnosis or staging. Report only morphological proof. Provide a concise summary of all findings."""

                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
                        ### INPUT DATA FOR CASE
                        1. IMAGE_OCT: [Attached B-scan]
                        2. IMAGE_SLO: [Attached SLO/Fundus image]
                        
                        ### YOUR ASSIGNMENT
                        Perform an inspection of the provided images as described above.
                        
                        ### REQUIRED OUTPUT STRUCTURE
                        You must conclude your report with the following formatted block:
                        
                        [EXECUTIVE SUMMARY]
                        - VISUAL FINDINGS
                        [/EXECUTIVE SUMMARY]
                        """
                    },
                    { "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_oct_image}"} },
                    { "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_slo_image}"} }
                ]
            }
        ]
        
        response = self.model_client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            temperature=0.2  
        )
        
        full_content = response.choices[0].message.content
        
        if "[EXECUTIVE SUMMARY]" in full_content:
            summary = full_content.split("[EXECUTIVE SUMMARY]")[-1].strip()
        else:
            summary = full_content 
          
        return {
            "agent": "Vision Specialist",
            "full_report": full_content,
            "summary": summary,
            "retfound_scores": retfound_scores,
            "mirage_scores": mirage_scores
        }