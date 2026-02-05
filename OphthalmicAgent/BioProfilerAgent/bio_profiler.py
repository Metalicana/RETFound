import os
from openai import AzureOpenAI
from data.loader import GenericEyeLoader
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

    def generate_narrative(self, metadata_dict):
        """
        Dynamically extracts available data and asks the LLM to write the profile.
        """
        # 1. Filter the dictionary to only include clinical info
        clinical_info = {
            k: v for k, v in metadata_dict.items() 
            if k not in self.internal_keys
        }

        # 2. Build the dynamic data string
        data_string = "\n".join([f"- {k.title()}: {v}" for k, v in clinical_info.items()])

        # 3. Create the prompt for the Bio-Profiler
        prompt = f"""
        You are a specialized Medical Bio-Profiler for an Ophthalmology Clinic.
        Below is raw metadata for a patient. Transform this data into a 
        concise, 3-sentence clinical narrative. 
        
        Highlight demographic risks or specific clinical values (like MD) 
        that a consultant should be aware of.

        PATIENT METADATA:
        {data_string}

        CLINICAL NARRATIVE:
        """
        
       # 4. Call the OpenAI API
        response = self.model_client.chat.completions.create(
            model="gpt-5.1",  
            messages=[
                {"role": "system", "content": "You are a professional medical scribe."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3  # Lower temperature for more factual, consistent medical summaries
        )

        return response.choices[0].message.content

# --- Test ---


## --- Integration Test Logic ---
#if __name__ == "__main__":
#    # 1. Setup paths
#    BASE_PATH = "/lustre/fs1/home/yu395012/OphthalmicAgent/"
#    
#    disease = 'DR'
#    loader = GenericEyeLoader(BASE_PATH)
#    df = loader.get_metadata(disease)
#    
#    # Grab the first test patient
#    test_rows = df[df['use'] == 'test']
#    if not test_rows.empty:
#        patient_record = loader.load_patient(disease, test_rows.iloc[0])
#        metadata = patient_record['metadata']
#    else:
#        print("No test data found to process.")
#        
#        
#        
#pprint(metadata)
#profiler = BioProfiler()
#print(profiler.generate_narrative(metadata))