import base64
import io
import os
from openai import AzureOpenAI
from PIL import Image
from dotenv import load_dotenv


load_dotenv() 
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

class DRSpecialist:

    def __init__(self, model_client=None):
    
      self.model_client = AzureOpenAI(
        azure_endpoint = endpoint, 
        api_key = api_key,
        api_version="2024-12-01-preview"
        )
      self.deployment_name = "gpt-5.1"

    def _prepare_image(self, numpy_img):
        """Converts numpy array from AgentState to Base64 for Azure."""
        pil_img = Image.fromarray(numpy_img.astype('uint8'))
        buff = io.BytesIO()
        pil_img.save(buff, format="JPEG")
        return base64.b64encode(buff.getvalue()).decode('utf-8')

    def analyze(self, state):
        narrative = state['clinical_narrative']
        base64_image = self._prepare_image(state['fundus_img'])
        
        # Multimodal Prompt: Text + Image
        messages = [
            {
                "role": "system", 
                "content": "You are a Senior Retina Specialist. Analyze the provided fundus image and patient narrative to detect Diabetic Retinopathy (DR)."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Patient Case Narrative:\n{narrative}\n\nPlease analyze the attached fundus photo for vascular abnormalities like microaneurysms or hemorrhages and provide a final DR assessment."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}
                    }
                ]
            }
        ]

        response = self.model_client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            temperature=0.2
        )
        
        return {
            "agent": "DR_Specialist",
            "decision": response.choices[0].message.content
        }