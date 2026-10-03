import os
from openai import AzureOpenAI
from pprint import pprint
from dotenv import load_dotenv

load_dotenv() 
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT = "gpt-5.1"
        
class BioProfiler:
    def __init__(self, model_client=None):
        
        self.model_client = AzureOpenAI(
          azure_endpoint = endpoint, 
          api_key = api_key,
          api_version="2024-12-01-preview"
          )

        # Keys that are for coding/logging, not clinical analysis
        self.internal_keys = ['filename', 'use', 'glaucoma', 'amd', 'dr']

    ## FOR GLAUCOMA SCREENING
    def generate_narrative(self, metadata_dict):
        """
        Dynamically extracts available data, completely ignoring MD scores and Ground Truth,
        while explicitly summarizing patient demographic baseline characteristics.
        """
        
        keys_to_use = [
            'age', 'gender', 'race', 'ethnicity'
        ]
        
        clinical_info = {
            k: v for k, v in metadata_dict.items() 
            if k.lower().strip() in keys_to_use
        }
        
        # 2. Build the dynamic data string
        data_string = "\n".join([f"- {k.title()}: {v}" for k, v in clinical_info.items()])

        # 3. Restructured prompt to explicitly demand demographic summaries safely
        prompt = f"""
        You are a specialized Medical Bio-Profiler for an Ophthalmology Clinic screening for Diabetic Retinopathy (DR).
        Below is raw metadata for a patient. Transform this data into a concise, 3-sentence clinical narrative. OCT scan and SLO image is present that will be provided to the Vision Agent.
        
        - MUST summarize the standard patient baseline details using available demographics (e.g., "A 60-year-old Hispanic male presents for diagnostic review...").
        - CRITICAL: Never present demographics (race, age, gender) as a direct cause or clinical proof of disease; treat them strictly as baseline descriptive metadata for the presentation case.
        
        PATIENT METADATA:
        {data_string}

        CLINICAL NARRATIVE:
        """
        
        # 4. Call the OpenAI API
        response = self.model_client.chat.completions.create(
            model=DEPLOYMENT,  
            messages=[
                {"role": "system", "content": "You are a professional medical scribe specializing in ophthalmic diseases."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3  
        )

        return response.choices[0].message.content