#################################################################################### AMD (to be) ##########################################################
#
##comment 1
#
#import sys
#import os
#import torch
#import numpy as np
#from torchvision import transforms
#from PIL import Image
#from openai import AzureOpenAI
#from dotenv import load_dotenv
#import base64
#import io
#from VisionAgent.linear_probing_oct3 import get_model_oct
#
#load_dotenv() 
#api_key = os.getenv("AZURE_OPENAI_API_KEY")
#endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
#
#class VisionSpecialistOct:
#    def __init__(self, path_oct, device=None):
#        
#        ### OCT MODEL ###
#        if device is None:
#            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#        else:
#            self.device = torch.device(device)
#
#        self.model_oct = get_model_oct()
#
#        if not os.path.exists(path_oct):
#            raise FileNotFoundError(f"Weights not found at {path_oct}")
#            
#        self.model_oct.load_state_dict(torch.load(path_oct, map_location=self.device, weights_only=True))
#        self.model_oct.to(self.device)
#        self.model_oct.eval()
#
#        self.model_client = AzureOpenAI(
#          azure_endpoint = endpoint, 
#          api_key = api_key,
#          api_version="2024-12-01-preview"
#        )
#        self.deployment_name = "gpt-5.1"
#        
#
#    def _preprocess_oct(self, oct_img):
##        image = Image.fromarray(oct_img).convert('RGB')
#        transform = transforms.Compose([
#            transforms.Resize((224, 224)),
#            transforms.ToTensor(),
#            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#        ])
##        img_tensor = transform(oct_img).unsqueeze(0).to(self.device) 
#
#        images = []
#
#        for oct_slice in oct_img:
#            img = Image.fromarray(oct_slice).convert("RGB")
#            img = transform(img)
#    
#            images.append(img)
#    
#        oct_tensor = torch.stack(images)
#    
#        return oct_tensor.unsqueeze(0).to(self.device)
#    #    
##        return img_tensor
#        
#    def get_features_oct(self, oct_img):
#        """
#        Runs the OCT slice through the single-node binary heads.
#        """
#        input_tensor = self._preprocess_oct(oct_img)
#        
#        with torch.no_grad():
##            print("MAYYYY AYAAAAA")
#            output = self.model_oct(input_tensor)
##            print("MAYYYY AYAAAAA 2")
#            # AMD head is now 1 single binary node [Batch, 1] instead of 3 nodes
#            amd_prob = torch.sigmoid(output['amd']).item()
#            dr_prob = torch.sigmoid(output['dr']).item()
#            glaucoma_prob = torch.sigmoid(output['glaucoma']).item()
#
#
#        return {
#            'AMD': {
#                'Probability': round(amd_prob, 4),
#                'Detected': int(amd_prob > 0.5)  # 0: Healthy, 1: Diseased
#            },
#            'DR': {
#                'Probability': round(dr_prob, 4),
#                'Detected': int(dr_prob > 0.5)
#            },
#            'Glaucoma': {
#                'Probability': round(glaucoma_prob, 4),
#                'Detected': int(glaucoma_prob > 0.5)
#            }
#        }
#          
#    def analyze(self, oct_img, state):
#        state["oct_diagnosis"] = self.get_features_oct(oct_img)
#    
#        # 2. Reformat data dictionaries for direct binary tracking strings
#        diag_key = "oct_diagnosis"
#        diag = state[diag_key]
#        for disease in ['AMD', 'DR', 'Glaucoma']:
#            prob_pct = diag[disease]['Probability'] * 100
#            state[diag_key][disease]['Prob_Pct'] = round(prob_pct, 2)
#            
#            # Assign simple binary classification strings
#            if disease == 'Glaucoma':
#                state[diag_key][disease]['Status'] = "High Risk" if prob_pct > 50 else "Low Risk"
#            else:
#                state[diag_key][disease]['Status'] = "Positive" if prob_pct > 50 else "Negative"
#
#        oct_data = state['oct_diagnosis']
#        retfound_scores = (
#            f"SOURCE: RETFound (OCT Imaging Specialist)\n"
##            f"AMD Probability: {oct_data['AMD']['Prob_Pct']}% -> Status: {oct_data['AMD']['Status']}\n" #CHANGE
#            f"DR Status: {oct_data['DR']['Status']} (Confidence: {oct_data['DR']['Prob_Pct']}%)\n"
##            f"Glaucoma Status: {oct_data['Glaucoma']['Status']} (Confidence: {oct_data['Glaucoma']['Prob_Pct']}%)\n"
#        )
#        
#        return retfound_scores
#        
#
#
#################################################################################### DR ##########################################################
#
##comment 1
#
#import sys
#import os
#import torch
#import numpy as np
#from torchvision import transforms
#from PIL import Image, ImageDraw, ImageFont
#from openai import AzureOpenAI
#from dotenv import load_dotenv
#import base64
#import io
#import cv2
#from VisionAgent.linear_probing_oct3 import get_model_oct
#
#load_dotenv() 
#api_key = os.getenv("AZURE_OPENAI_API_KEY")
#endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
#
#class VisionSpecialistOct:
#    def __init__(self, path_oct, device=None):
#        
#        ### OCT MODEL ###
#        if device is None:
#            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#        else:
#            self.device = torch.device(device)
#
#        self.model_oct = get_model_oct()
#
#        if not os.path.exists(path_oct):
#            raise FileNotFoundError(f"Weights not found at {path_oct}")
#            
#        self.model_oct.load_state_dict(torch.load(path_oct, map_location=self.device, weights_only=True))
#        self.model_oct.to(self.device)
#        self.model_oct.eval()
#
#        self.model_client = AzureOpenAI(
#          azure_endpoint = endpoint, 
#          api_key = api_key,
#          api_version="2024-12-01-preview"
#        )
#        self.deployment_name = "gpt-5.1"
#        
#
#    def _preprocess_oct(self, oct_img):
##        image = Image.fromarray(oct_img).convert('RGB')
#        transform = transforms.Compose([
#            transforms.Resize((224, 224)),
#            transforms.ToTensor(),
#            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#        ])
##        img_tensor = transform(oct_img).unsqueeze(0).to(self.device) 
#
#        images = []
#
#        for oct_slice in oct_img:
#            img = Image.fromarray(oct_slice).convert("RGB")
#            img = transform(img)
#    
#            images.append(img)
#    
#        oct_tensor = torch.stack(images)
#    
#        return oct_tensor.unsqueeze(0).to(self.device)
#    #    
##        return img_tensor
#        
#    def get_features_oct(self, oct_img):
#        """
#        Runs the OCT slice through the single-node binary heads.
#        """
#        input_tensor = self._preprocess_oct(oct_img)
#        
#        with torch.no_grad():
##            print("MAYYYY AYAAAAA")
#            output = self.model_oct(input_tensor)
##            print("MAYYYY AYAAAAA 2")
#            # AMD head is now 1 single binary node [Batch, 1] instead of 3 nodes
#            amd_prob = torch.sigmoid(output['amd']).item()
#            dr_prob = torch.sigmoid(output['dr']).item()
#            glaucoma_prob = torch.sigmoid(output['glaucoma']).item()
#
#
#        return {
#            'AMD': {
#                'Probability': round(amd_prob, 4),
#                'Detected': int(amd_prob > 0.5)  # 0: Healthy, 1: Diseased
#            },
#            'DR': {
#                'Probability': round(dr_prob, 4),
#                'Detected': int(dr_prob > 0.5)
#            },
#            'Glaucoma': {
#                'Probability': round(glaucoma_prob, 4),
#                'Detected': int(glaucoma_prob > 0.5)
#            }
#        }
##    def _create_oct_montage(self, oct_volume, middle_slice):
##        """
##        oct_volume: (8,H,W)
##        returns PIL image
##        """
##    
##        slices = []
##    
##        for s in oct_volume:
##            if s.max() <= 1:
##                s = (s * 255).astype(np.uint8)
##            else:
##                s = s.astype(np.uint8)
##    
##            slices.append(s)
##    
##        top = np.concatenate(slices[:4], axis=1)
##        bottom = np.concatenate(slices[4:], axis=1)
##    
##        montage = np.concatenate([top, bottom], axis=0)
##        montage = Image.fromarray(montage)
##        
##        print(f"montage size: ", montage.size)
##        
##        montage.save("oct_montage.png")
##        
##        return montage
#        
##    def _create_clahe_montage(self, oct_volume, middle_slice):
##      clahe = cv2.createCLAHE(
##          clipLimit=1.5,
##          tileGridSize=(8,8)
##      )
##  
##      enhanced = []
##  
##      for s in oct_volume:
##  
##          if s.max() <= 1:
##              s = (s*255).astype(np.uint8)
##          else:
##              s = s.astype(np.uint8)
##  
##          enhanced.append(clahe.apply(s))
##  
##      top = np.concatenate(enhanced[:4], axis=1)
##      bottom = np.concatenate(enhanced[4:], axis=1)
##  
##      montage = np.concatenate([top,bottom],axis=0)
##  
##      montage = Image.fromarray(montage)
##      montage.save("oct_clahe_montage.png")    
##      
##      return montage
#    def _create_clahe_montage(self, oct_volume, middle_slice):
#  
#      clahe = cv2.createCLAHE(
#          clipLimit=1.5,
#          tileGridSize=(8,8)
#      )
#
#      # ---------- middle slice ----------
#      if middle_slice.max() <= 1:
#          middle = (middle_slice * 255).astype(np.uint8)
#      else:
#          middle = middle_slice.astype(np.uint8)
#  
#      middle = clahe.apply(middle)
#  
#      H, W = middle.shape
#  
#      middle_large = cv2.resize(
#          middle,
#          (W*2, H*2),
#          interpolation=cv2.INTER_CUBIC
#      )
#  
#      # ---------- choose only 4 context slices ----------
#      context_ids = [2, 3, 4, 5]
#  
#      context = []
#  
#      for idx in context_ids:
#  
#          s = oct_volume[idx]
#  
#          if s.max() <= 1:
#              s = (s*255).astype(np.uint8)
#          else:
#              s = s.astype(np.uint8)
#  
#          s = clahe.apply(s)
#  
#          context.append(s)
#  
#      GAP = 12
#  
#      v_gap = np.ones((H, GAP), dtype=np.uint8) * 255
#  
#      bottom = np.concatenate(
#          [
#              context[0], v_gap,
#              context[1], v_gap,
#              context[2], v_gap,
#              context[3]
#          ],
#          axis=1
#      )
#  
#      # ---------- center the large image ----------
#      total_width = bottom.shape[1]
#      
#      TITLE_H = 35
#
#      title1 = np.ones((TITLE_H, total_width), dtype=np.uint8) * 255
#      title2 = np.ones((TITLE_H, total_width), dtype=np.uint8) * 255
#      
#      pad_left = (total_width - middle_large.shape[1]) // 2
#      pad_right = total_width - middle_large.shape[1] - pad_left
#  
#      middle_large = np.pad(
#          middle_large,
#          ((0,0),(pad_left,pad_right)),
#          mode="constant",
#          constant_values=255
#      )
#  
#      sep = np.ones((20, total_width), dtype=np.uint8) * 255
#  
#      montage = np.concatenate(
#          [
#            title1,
#            middle_large,
#            sep,
#            title2,
#            bottom
#          ],
#          axis=0
#      )
#  
#      montage = Image.fromarray(montage)
#      draw = ImageDraw.Draw(montage)
#
#      try:
#          font = ImageFont.truetype( "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
#      except:
#          font = ImageFont.load_default()
#      
#      draw.text((20, 8), "Representative OCT Slice", fill=0, font=font)
#      
#      draw.text(
#          (20, TITLE_H + middle_large.shape[0] + sep.shape[0] + 8),
#          "Adjacent OCT Slices",
#          fill=0,
#          font=font,
#      )
##      montage.save("oct_clahe_montage.png")
#  
#      return montage
#      
#    def _prepare_image(self, pil_img):
#  
#      buff = io.BytesIO()
#  
#      pil_img.save(buff, format="JPEG")
#  
#      return base64.b64encode(
#          buff.getvalue()
#      ).decode()  
#           
#    def analyze(self, oct_img, middle_slice, state):
#        state["oct_diagnosis"] = self.get_features_oct(oct_img) 
#        
##        ##only middle slice experiment
##        if middle_slice.max() <= 1:
##          middle = (middle_slice * 255).astype(np.uint8)
##        else:
##          middle = middle_slice.astype(np.uint8)
##        montage = Image.fromarray(middle)
#        
#        montage = self._create_clahe_montage(oct_img, middle_slice)
##        self._save_clahe_montage(oct_img)
#        base64_oct = self._prepare_image(montage)
#         
#        
##        You are given one representative central OCT B-scan (displayed as the large image) along with four adjacent B-scans from the same OCT volume (displayed below).
##You are given one representative central OCT B-scan.        
#        messages_oct = [
#        {
#        "role":"system",
#        "content":
#        """
#        You are an ophthalmic image analysis specialist reviewing OCT B-scans.
#        
#        The OCT classifier was uncertain about this case. Your task is to provide objective visual observations that may help another AI agent determine whether diabetic retinopathy (DR) is present. You are given one representative central OCT B-scan (displayed as the large image) along with four adjacent B-scans from the same OCT volume (displayed below).
#        
#        Important instructions:
#        
#        * Do not provide a final diagnosis.
#        * Do not estimate the probability of DR.
#        * Base your observations only on visible OCT findings.
#        
#        Focus on:
#        
#        * Overall retinal layer organization.
#        * Foveal contour.
#        * Retinal thickening or thinning.
#        * Intraretinal cystoid spaces if visible.
#        * Hyperreflective foci.
#        * Fluid accumulation.
#        * Distortion of retinal architecture.
#        * Any other abnormalities visible on OCT.
#        
#        Structure your response as:
#    
#        DR-Relevant Features:
#        ...
#        
#        Overall Impression:
#        Provide a brief summary of your findings related to diabetic retinopathy.       
#        """
#        },
#        {
#        "role":"user",
#        "content":[
#        {
#        "type":"text",
#        "text":"Analyze the following OCT."
#        },
#        {
#        "type":"image_url",
#        "image_url":{
#        "url":f"data:image/jpeg;base64,{base64_oct}"
#        }
#        }
#        ]
#        }
#        ]
#        
#        response = self.model_client.chat.completions.create(
#
#            model=self.deployment_name,
#        
#            messages=messages_oct,
#        
#            max_completion_tokens=500
#        )
#        
#        oct_report = response.choices[0].message.content
#
#        # 2. Reformat data dictionaries for direct binary tracking strings
#        diag_key = "oct_diagnosis"
#        diag = state[diag_key]
#        for disease in ['AMD', 'DR', 'Glaucoma']:
#            prob_pct = diag[disease]['Probability'] * 100
#            state[diag_key][disease]['Prob_Pct'] = round(prob_pct, 2)
#            
#            # Assign simple binary classification strings
#            if disease == 'Glaucoma':
#                state[diag_key][disease]['Status'] = "High Risk" if prob_pct > 50 else "Low Risk"
#            else:
#                state[diag_key][disease]['Status'] = "Positive" if prob_pct > 50 else "Negative"
#
#        oct_data = state['oct_diagnosis']
#        retfound_scores = (
#            f"SOURCE: RETFound (OCT Imaging Specialist)\n"
##            f"AMD Probability: {oct_data['AMD']['Prob_Pct']}% -> Status: {oct_data['AMD']['Status']}\n" #CHANGE
#            f"DR Status: {oct_data['DR']['Status']} (Confidence: {oct_data['DR']['Prob_Pct']}%)\n"
##            f"Glaucoma Status: {oct_data['Glaucoma']['Status']} (Confidence: {oct_data['Glaucoma']['Prob_Pct']}%)\n"
#        )
#        
##        print(oct_report)
#        return retfound_scores, oct_report





################################################################################### Glaucoma ##########################################################

#comment 1

import sys
import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from openai import AzureOpenAI
from dotenv import load_dotenv
import base64
import io
import cv2
from VisionAgent.linear_probing_oct3 import get_model_oct

load_dotenv() 
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

class VisionSpecialistOct:
    def __init__(self, path_oct, device=None):
        
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

        self.model_client = AzureOpenAI(
          azure_endpoint = endpoint, 
          api_key = api_key,
          api_version="2024-12-01-preview"
        )
        self.deployment_name = "gpt-5.1"
        

    def _preprocess_oct(self, oct_img):
#        image = Image.fromarray(oct_img).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
#        img_tensor = transform(oct_img).unsqueeze(0).to(self.device) 

        images = []

        for oct_slice in oct_img:
            img = Image.fromarray(oct_slice).convert("RGB")
            img = transform(img)
    
            images.append(img)
    
        oct_tensor = torch.stack(images)
    
        return oct_tensor.unsqueeze(0).to(self.device)
    #    
#        return img_tensor
        
    def get_features_oct(self, oct_img):
        """
        Runs the OCT slice through the single-node binary heads.
        """
        input_tensor = self._preprocess_oct(oct_img)
        
        with torch.no_grad():
#            print("MAYYYY AYAAAAA")
            output = self.model_oct(input_tensor)
#            print("MAYYYY AYAAAAA 2")
            # AMD head is now 1 single binary node [Batch, 1] instead of 3 nodes
            amd_prob = torch.sigmoid(output['amd']).item()
            dr_prob = torch.sigmoid(output['dr']).item()
            glaucoma_prob = torch.sigmoid(output['glaucoma']).item()


        return {
            'AMD': {
                'Probability': round(amd_prob, 4),
                'Detected': int(amd_prob > 0.5)  # 0: Healthy, 1: Diseased
            },
            'DR': {
                'Probability': round(dr_prob, 4),
                'Detected': int(dr_prob > 0.5)
            },
            'Glaucoma': {
                'Probability': round(glaucoma_prob, 4),
                'Detected': int(glaucoma_prob > 0.5)
            }
        }
#    def _create_oct_montage(self, oct_volume, middle_slice):
#        """
#        oct_volume: (8,H,W)
#        returns PIL image
#        """
#    
#        slices = []
#    
#        for s in oct_volume:
#            if s.max() <= 1:
#                s = (s * 255).astype(np.uint8)
#            else:
#                s = s.astype(np.uint8)
#    
#            slices.append(s)
#    
#        top = np.concatenate(slices[:4], axis=1)
#        bottom = np.concatenate(slices[4:], axis=1)
#    
#        montage = np.concatenate([top, bottom], axis=0)
#        montage = Image.fromarray(montage)
#        
#        print(f"montage size: ", montage.size)
#        
#        montage.save("oct_montage.png")
#        
#        return montage
        
#    def _create_clahe_montage(self, oct_volume, middle_slice):
#      clahe = cv2.createCLAHE(
#          clipLimit=1.5,
#          tileGridSize=(8,8)
#      )
#  
#      enhanced = []
#  
#      for s in oct_volume:
#  
#          if s.max() <= 1:
#              s = (s*255).astype(np.uint8)
#          else:
#              s = s.astype(np.uint8)
#  
#          enhanced.append(clahe.apply(s))
#  
#      top = np.concatenate(enhanced[:4], axis=1)
#      bottom = np.concatenate(enhanced[4:], axis=1)
#  
#      montage = np.concatenate([top,bottom],axis=0)
#  
#      montage = Image.fromarray(montage)
#      montage.save("oct_clahe_montage.png")    
#      
#      return montage
    def _create_clahe_montage(self, oct_volume, middle_slice):
  
      clahe = cv2.createCLAHE(
          clipLimit=1.5,
          tileGridSize=(8,8)
      )

      # ---------- middle slice ----------
      if middle_slice.max() <= 1:
          middle = (middle_slice * 255).astype(np.uint8)
      else:
          middle = middle_slice.astype(np.uint8)
  
      middle = clahe.apply(middle)
  
      H, W = middle.shape
  
      middle_large = cv2.resize(
          middle,
          (W*2, H*2),
          interpolation=cv2.INTER_CUBIC
      )
  
      # ---------- choose only 4 context slices ----------
      context_ids = [2, 3, 4, 5]
  
      context = []
  
      for idx in context_ids:
  
          s = oct_volume[idx]
  
          if s.max() <= 1:
              s = (s*255).astype(np.uint8)
          else:
              s = s.astype(np.uint8)
  
          s = clahe.apply(s)
  
          context.append(s)
  
      GAP = 12
  
      v_gap = np.ones((H, GAP), dtype=np.uint8) * 255
  
      bottom = np.concatenate(
          [
              context[0], v_gap,
              context[1], v_gap,
              context[2], v_gap,
              context[3]
          ],
          axis=1
      )
  
      # ---------- center the large image ----------
      total_width = bottom.shape[1]
      
      TITLE_H = 35

      title1 = np.ones((TITLE_H, total_width), dtype=np.uint8) * 255
      title2 = np.ones((TITLE_H, total_width), dtype=np.uint8) * 255
      
      pad_left = (total_width - middle_large.shape[1]) // 2
      pad_right = total_width - middle_large.shape[1] - pad_left
  
      middle_large = np.pad(
          middle_large,
          ((0,0),(pad_left,pad_right)),
          mode="constant",
          constant_values=255
      )
  
      sep = np.ones((20, total_width), dtype=np.uint8) * 255
  
      montage = np.concatenate(
          [
            title1,
            middle_large,
            sep,
            title2,
            bottom
          ],
          axis=0
      )
  
      montage = Image.fromarray(montage)
      draw = ImageDraw.Draw(montage)

      try:
          font = ImageFont.truetype( "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
      except:
          font = ImageFont.load_default()
      
      draw.text((20, 8), "Representative OCT Slice", fill=0, font=font)
      
      draw.text(
          (20, TITLE_H + middle_large.shape[0] + sep.shape[0] + 8),
          "Adjacent OCT Slices",
          fill=0,
          font=font,
      )
#      montage.save("oct_clahe_montage.png")
  
      return montage
      
    def _prepare_image(self, pil_img):
  
      buff = io.BytesIO()
  
      pil_img.save(buff, format="JPEG")
  
      return base64.b64encode(
          buff.getvalue()
      ).decode()  
           
    def analyze(self, oct_img, middle_slice, state):
        state["oct_diagnosis"] = self.get_features_oct(oct_img) 
        
        ##only middle slice experiment
        if middle_slice.max() <= 1:
          middle = (middle_slice * 255).astype(np.uint8)
        else:
          middle = middle_slice.astype(np.uint8)
        montage = Image.fromarray(middle)
        
#        montage = self._create_clahe_montage(oct_img, middle_slice)
#        self._save_clahe_montage(oct_img)
        base64_oct = self._prepare_image(montage)
         
        
#        You are given one representative central OCT B-scan (displayed as the large image) along with four adjacent B-scans from the same OCT volume (displayed below).
#You are given one representative central OCT B-scan.        
        messages_oct = [
        {
        "role":"system",
        "content":
      """
You are an ophthalmic image analysis specialist reviewing OCT B-scans.

The OCT classifier was uncertain about this case. Your task is to provide objective visual observations that may help another AI agent determine whether glaucoma is present. You are given one representative central OCT B-scan (displayed as the large image) along with four adjacent B-scans from the same OCT volume (displayed below).

Important instructions:

* Do not provide a final diagnosis.
* Do not estimate the probability of glaucoma.
* Base your observations only on visible OCT findings.

Focus on:

* Overall retinal layer organization.
* Foveal contour.
* Thickness and continuity of the inner retinal layers.
* Any visible thinning or loss of retinal tissue.
* Localized structural abnormalities that appear consistent across adjacent slices.
* Asymmetry or irregularity visible within the provided scans.
* Any other structural findings that may be relevant.

Do NOT comment on:

* Cup-to-disc ratio.
* Neuroretinal rim thickness.
* Optic disc appearance.
* Retinal nerve fiber layer measurements outside the visible scan.
* Findings that are not directly observable in these OCT B-scans.

Structure your response as:

Glaucoma-Relevant Features:
...

Overall Impression:
Provide a brief summary of your structural observations.
"""
        },
        {
        "role":"user",
        "content":[
        {
        "type":"text",
        "text":"Analyze the following OCT."
        },
        {
        "type":"image_url",
        "image_url":{
        "url":f"data:image/jpeg;base64,{base64_oct}"
        }
        }
        ]
        }
        ]
        
        response = self.model_client.chat.completions.create(

            model=self.deployment_name,
        
            messages=messages_oct,
        
            max_completion_tokens=500
        )
        
        oct_report = response.choices[0].message.content

        # 2. Reformat data dictionaries for direct binary tracking strings
        diag_key = "oct_diagnosis"
        diag = state[diag_key]
        for disease in ['AMD', 'DR', 'Glaucoma']:
            prob_pct = diag[disease]['Probability'] * 100
            state[diag_key][disease]['Prob_Pct'] = round(prob_pct, 2)
            
            # Assign simple binary classification strings
            if disease == 'Glaucoma':
                state[diag_key][disease]['Status'] = "High Risk" if prob_pct > 50 else "Low Risk"
            else:
                state[diag_key][disease]['Status'] = "Positive" if prob_pct > 50 else "Negative"

        oct_data = state['oct_diagnosis']
        retfound_scores = (
            f"SOURCE: RETFound (OCT Imaging Specialist)\n"
#            f"AMD Probability: {oct_data['AMD']['Prob_Pct']}% -> Status: {oct_data['AMD']['Status']}\n" #CHANGE
#            f"DR Status: {oct_data['DR']['Status']} (Confidence: {oct_data['DR']['Prob_Pct']}%)\n"
            f"Glaucoma Status: {oct_data['Glaucoma']['Status']} (Confidence: {oct_data['Glaucoma']['Prob_Pct']}%)\n"
        )
        
#        print(oct_report)
        return retfound_scores, oct_report