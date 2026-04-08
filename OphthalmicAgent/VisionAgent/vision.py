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
    
        base64_slo_image = self._prepare_image(state['fundus_img'])
        base64_oct_image = self._prepare_image(state['oct_img'])
        
        oct_diagnosis = state['oct_diagnosis']
        slo_diagnosis = state['slo_diagnosis']
        
        if oct_diagnosis['AMD']['Advanced'] > 0.5:
          amd_description_oct = "Advanced/Wet AMD, STAGE 3"
          amd_probability_oct = oct_diagnosis['AMD']['Advanced'] * 100
          
        elif oct_diagnosis['AMD']['Intermediate'] > 0.5:
          amd_description_oct = "Intermediate Dry AMD, STAGE 2"
          amd_probability_oct = oct_diagnosis['AMD']['Intermediate'] * 100
          
        elif oct_diagnosis['AMD']['Early'] > 0.5:
          amd_description_oct = "Early signs of AMD detected, STAGE 1"
          amd_probability_oct = oct_diagnosis['AMD']['Early'] * 100
          
        else:
          amd_description_oct = "No significant signs of AMD, STAGE 0"
          amd_probability_oct = 50
          
        dr_status_oct = "Positive" if oct_diagnosis['DR']['Probability'] > 0.5 else "Negative"
        dr_probability_oct = oct_diagnosis['DR']['Probability'] * 100
        
        gl_status_oct = "High Risk" if oct_diagnosis['Glaucoma']['Probability'] > 0.5 else "Low Risk"
        gl_probability_oct = oct_diagnosis['Glaucoma']['Probability'] * 100
        
        
        if slo_diagnosis['AMD']['Advanced'] > 0.5:
          amd_description_slo = "Advanced/Wet AMD, STAGE 3"
          amd_probability_slo = slo_diagnosis['AMD']['Advanced'] * 100
          
        elif slo_diagnosis['AMD']['Intermediate'] > 0.5:
          amd_description_slo = "Intermediate Dry AMD, STAGE 2"
          amd_probability_slo = slo_diagnosis['AMD']['Intermediate'] * 100
          
        elif slo_diagnosis['AMD']['Early'] > 0.5:
          amd_description_slo = "Early signs of AMD detected, STAGE 1"
          amd_probability_slo = slo_diagnosis['AMD']['Early'] * 100
          
        else:
          amd_description_slo = "No significant signs of AMD, STAGE 0"
          amd_probability_slo = 50
          
        dr_status_slo = "Positive" if slo_diagnosis['DR']['Probability'] > 0.5 else "Negative"
        dr_probability_slo = slo_diagnosis['DR']['Probability'] * 100
        
        gl_status_slo = "High Risk" if slo_diagnosis['Glaucoma']['Probability'] > 0.5 else "Low Risk"
        gl_probability_slo = slo_diagnosis['Glaucoma']['Probability'] * 100

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Senior Ophthalmic Imaging Specialist. Your task is to produce a 'Verification Summary' "
                    "that correlates AI probabilities with visual evidence from SLO and OCT images.\n\n"
                    
                    "STRUCTURE YOUR SUMMARY AS FOLLOWS:\n"
                    "1. NUMERICAL DATA: Repeat the raw probabilities for AMD, DR, and Glaucoma.\n" 
                    "2. VISUAL AUDIT: Describe specific markers (drusen, hemmorrhages, disc cupping) seen in the images. If you visually identify any "                       "hyper-reflective bumps or elevations at the RPE-Bruch's membrane interface in the OCT B-scan you MUST flag this as 'Visual Evidence                       of AMD'.\n"                
                    "3. ALIGNMENT STATUS: Explicitly state 'Agree' or 'Conflict' for each disease.\n"
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
                        ### Inputs for Audit
                        **RETFound (OCT):** AMD: {amd_description_oct} (p={amd_probability_oct:.2f}), DR: {dr_status_oct} (p={dr_probability_oct:.2f}), Glaucoma: {gl_status_oct} (p={gl_probability_oct:.2f})
                        **MIRAGE (SLO):** AMD: {amd_description_slo} (p={amd_probability_slo:.2f}), DR: {dr_status_slo} (p={dr_probability_slo:.2f}), Glaucoma: {gl_status_slo} (p={gl_probability_slo:.2f})
        
                        ### ASSIGNMENT
                        1. Inspect the images. Do you see the physical lesions that justify the p-values?
                        2. Pay special attention to the sub-retinal space in the OCT - if you see even tiny bumps, flag it as evidence for AMD.
                        3. Create the [EXECUTIVE SUMMARY] containing: 
                           - The raw p-values.
                           - Your visual findings.
                           - An 'Alignment' statement (Agree/Conflict).
                        """
                    },
                    { "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_slo_image}"} },
                    { "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_oct_image}"} }
                ]
            }
        ]
        
        print("\n\n--- Sending Visual Data to Vision Specialist ---")
        
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
            "summary": summary
        }