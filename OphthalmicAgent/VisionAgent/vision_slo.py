#comment 1

import sys
import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageOps
from openai import AzureOpenAI
from dotenv import load_dotenv
import base64
import io
import cv2
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from scipy.ndimage import find_objects
import matplotlib.pyplot as plt

load_dotenv() 
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

slo_healthy = Image.open("slo.png")
slo_healthy_array = np.array(slo_healthy)

class VisionSpecialistSlo:
    def __init__(self, path_oct, path_slo, device=None):
        
        self.model_client = AzureOpenAI(
          azure_endpoint = endpoint, 
          api_key = api_key,
          api_version="2024-12-01-preview"
        )
        self.deployment_name = "gpt-5.1"
        
#        #CUP TO DISC RATIO
        self.cdr_model_name = "pamixsun/segformer_for_optic_disc_cup_segmentation"
        self.processor_cdr = SegformerImageProcessor.from_pretrained(self.cdr_model_name)
        self.model_cdr = SegformerForSemanticSegmentation.from_pretrained(self.cdr_model_name)
        
        self.global_var = 0
        
    def _enhance_image(self, numpy_img):
      # Convert to uint8 if needed
      if numpy_img.dtype != np.uint8:    
          numpy_img = numpy_img.astype(np.uint8)
  
      # Apply CLAHE
      clahe = cv2.createCLAHE(
          clipLimit=1.5,
          tileGridSize=(8, 8)
      )
      
      enhanced = clahe.apply(numpy_img)
#      Image.fromarray(enhanced).save(f"clahe_{self.global_var}.png")
      self.global_var += 1
      return enhanced
  
    def segment_cup_and_disc(self, enhanced_slo):
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.model_cdr.to(device)
      self.model_cdr.eval()
      
      raw_image = Image.fromarray(enhanced_slo).convert("RGB")
      raw_image = ImageOps.equalize(raw_image)
      
#       3. Run inference
      inputs = self.processor_cdr(images=raw_image, return_tensors="pt").to(device)
      with torch.no_grad():
          outputs = self.model_cdr(**inputs)
          logits = outputs.logits  # shape: (batch_size, num_labels, height/4, width/4)
      
      # 4. Upsample logits to match original image size
      upsampled_logits = torch.nn.functional.interpolate(
          logits,
          size=raw_image.size[::-1], # (height, width)
          mode="bilinear",
          align_corners=False
      )
      
      # 5. Get argmax to find predicted classes
      # Typical class mapping for this dataset: 0 = Background, 1 = Optic Disc, 2 = Optic Cup
      prediction = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
      
      v_cdr, _ = self.calculate_cdr_from_mask(prediction)
#      self.visualize_results(raw_image, prediction)
      
      return v_cdr
#      return 1
            
    def calculate_cdr_from_mask(self, prediction_mask):
      """
      Calculates the Vertical and Horizontal Cup-to-Disc Ratio (CDR) from a mask.
      Assumes:
        1 = Optic Disc
        2 = Optic Cup
      """
      # 1. Create binary masks for the Disc and the Cup
      # Note: The disc mask must include the cup area because biologically 
      # the cup sits inside the disc.
      disc_mask = (prediction_mask == 1) | (prediction_mask == 2)
      cup_mask = (prediction_mask == 2)
      
      # Check if both structures were actually detected
      if not np.any(disc_mask) or not np.any(cup_mask):
          return -1, -1
  
      # 2. Find the bounding boxes of both masks using scipy
      # find_objects returns a tuple of slice objects corresponding to the bounding box
      disc_box = find_objects(disc_mask)[0]
      cup_box = find_objects(cup_mask)[0]
      
      # 3. Calculate heights (vertical pixels) and widths (horizontal pixels)
      # slice(start, stop, None) -> length = stop - start
      disc_height = disc_box[0].stop - disc_box[0].start
      disc_width = disc_box[1].stop - disc_box[1].start
      
      cup_height = cup_box[0].stop - cup_box[0].start
      cup_width = cup_box[1].stop - cup_box[1].start
      
      # 4. Calculate Ratios
      v_cdr = cup_height / disc_height
      h_cdr = cup_width / disc_width
      
      v_cdr = round(v_cdr,3)
      
      return v_cdr, h_cdr  
    
    def visualize_results(self, original_image, mask):
      fig, axes = plt.subplots(1, 3, figsize=(15, 5))
      
      # Display Original Image
      axes[0].imshow(original_image)
      axes[0].set_title("Pre-processed SLO Image")
      axes[0].axis("off")
      
      # Display Mask 
      # (0: Background=Black, 1: Disc=Red/Gray, 2: Cup=White/Yellow depending on cmap)
      axes[1].imshow(mask, cmap="gray")
      axes[1].set_title("Raw Segmentation Mask")
      axes[1].axis("off")
      
      # Overlay Mask onto Original Image
      axes[2].imshow(original_image)
      # Mask values: create an alpha channel overlay
      overlay = np.zeros((*mask.shape, 4))
      overlay[mask == 1] = [1, 0, 0, 0.4]  # Translucent Red for Optic Disc
      overlay[mask == 2] = [0, 0, 1, 0.5]  # Translucent Blue for Optic Cup
      
      axes[2].imshow(overlay)
      axes[2].set_title("Overlay (Red=Disc, Blue=Cup)")
      axes[2].axis("off")
      
      plt.tight_layout()
  #    plt.show()
      plt.savefig(f"cdr_output{self.global_var-1}.png")
        
    def _prepare_image(self, numpy_img): 
      # Convert to PIL
      pil_img = Image.fromarray(numpy_img).convert("RGB")
      buff = io.BytesIO()
      pil_img.save(buff, format="JPEG")
      
#      Image.fromarray(numpy_img).save("clahe_cropped.png")
#      Image.fromarray(enhanced).save("clahe_cropped.png")
  
      return base64.b64encode(buff.getvalue()).decode("utf-8")
      
    def analyze(self, fundus_img, state): 
        enhanced_slo = self._enhance_image(fundus_img)
        v_cdr = self.segment_cup_and_disc(enhanced_slo)
               
        base64_slo_image = self._prepare_image(enhanced_slo)
#        base64_slo_healthy = self._prepare_image(slo_healthy_array)
        print(f"CDR is: {v_cdr}")
        
        # Brief clinical context output
        if v_cdr > 0.7:
            print("Note: High vertical CDR detected (> 0.7). Potential glaucomatous risk.")
        elif v_cdr > 0.4:
            print("Note: Moderate vertical CDR detected (0.4 - 0.7). Check for asymmetry.")
        elif v_cdr == -1:
            print("Error: Could not find both optic disc and cup in the mask.")
            v_cdr = "Not Available"
        else:
            print("Note: Normal vertical CDR detected (< 0.4).") 
            
        messages_slo = [
            {
                "role": "system",
                "content": (
                
                #GLAUCOMA
                
                    """You are an ophthalmic image analysis specialist reviewing a monochrome Scanning Laser Ophthalmoscopy (SLO) image.

                    The OCT-based diagnostic model was uncertain about this case. Your role is to provide additional visual evidence that may help another AI agent determine whether glaucoma is present.
                    
                    Important instructions:
                    
                    * Do not provide a final diagnosis.
                    * Do not provide a final glaucoma probability.
                    * Base your assessment only on visible image features.
                    * Prefer cautious interpretation over certainty, but do not avoid making observations simply because they are imperfect.
                    
                    Focus on:
                    
                    * Appearance of the optic disc region.
                    * Vessel configuration around the optic disc.
                    * Structural patterns that appear unusual or potentially suspicious.
                    * Features that appear normal or reassuring.
                    * Findings that increase or decrease concern for glaucoma.
                    
                    Your goal is not to diagnose glaucoma, but to identify image features that may support or weaken suspicion of glaucoma.
                    
                    Structure your response as:

                    Glaucoma-Relevant Features:
                    ...
                    
                    Overall Impression:
                    Provide a brief summary of your findings related to glaucoma diagnosis.
                    """

## AMD
# """                   You are an ophthalmic image analysis specialist reviewing a monochrome Scanning Laser Ophthalmoscopy (SLO) image.
#
#The OCT-based diagnostic model was uncertain about this case. Your role is to provide additional visual evidence that may help another AI agent determine whether Age-related Macular Degeneration (AMD) is present.
#
#Important instructions:
#
#* Do not provide a final diagnosis.
#* Do not provide a final AMD probability.
#* Base your assessment only on visible image features.
#* Prefer cautious interpretation over certainty, but do not avoid making observations simply because they are imperfect.
#
#Focus on:
#
#* Appearance of the macular region.
#* Overall retinal texture and reflectance patterns.
#* Any visible abnormalities within or around the macula.
#* Areas that appear unusually bright, dark, irregular, or structurally altered.
#* Any signs of atrophy, pigmentary irregularity, deposits, or other macular abnormalities that are visually apparent.
#* Features that appear normal or reassuring.
#
#Your goal is not to diagnose AMD, but to identify image features that may support or weaken suspicion of AMD.
#
#Structure your response as:
#
#AMD-Relevant Features:
#...
#
#Overall Impression:
#Provide a brief summary of your findings related to AMD diagnosis.
#"""


##DR
#"""
#You are an ophthalmic image analysis specialist reviewing a monochrome Scanning Laser Ophthalmoscopy (SLO) image.
#
#The OCT-based diagnostic model was uncertain about this case. Your role is to provide additional visual evidence that may help another AI agent determine whether Diabetic Retinopathy (DR) is present. 
#
#Important instructions:
#
#* Do not provide a final diagnosis.
#* Do not provide a final DR probability.
#* Base your assessment only on visible image features.
#* Prefer cautious interpretation over certainty, but do not avoid making observations simply because they are imperfect.
#
#Focus on:
#
#* Overall retinal appearance.
#* Visibility and configuration of retinal blood vessels.
#* Any visible abnormalities in the retinal vasculature.
#* Areas that appear unusually bright, dark, irregular, or structurally altered.
#* Any visually apparent lesions, hemorrhage-like regions, exudate-like regions, vessel irregularities, or other retinal abnormalities.
#* Features that appear normal or reassuring.
#
#Your goal is not to diagnose Diabetic Retinopathy, but to identify image features that may support or weaken suspicion of DR.
#
#Structure your response as:
#
#DR-Relevant Features:
#...
#
#Overall Impression:
#Provide a brief summary of your findings related to diabetic retinopathy.

#"""

                )
            },
            {
                "role": "user",
                "content": [
#                    {
#                        "type": "text",
#                        "text": "Reference image of healthy SLO"
#                    },
#                    {
#                        "type": "image_url",
#                        "image_url": {
#                            "url": f"data:image/jpeg;base64,{base64_slo_healthy}"
#                        }
#                    },
                    
                    {
                        "type": "text",
                        "text": "Analyze the following monochrome SLO image and provide objective visual observations."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_slo_image}"
                        }
                    }
                ]
            }
        ]
        
        response = self.model_client.chat.completions.create(
            model=self.deployment_name,
            messages=messages_slo,
            temperature=0.2  
        )
        
        full_content = response.choices[0].message.content
          
        return full_content, v_cdr