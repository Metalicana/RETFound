import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from pprint import pprint

class GenericEyeLoader:
    def __init__(self, base_path): 
        """
        base_path: The root folder containing DR, AMD, and Glaucoma folders.
        """
        self.base_path = base_path

    def get_metadata(self, disease):
        """Loads the correct CSV based on disease name."""
        # Standardizing naming: e.g., 'glaucoma' -> 'data_summary_glaucoma.csv'
        csv_name = f"data_summary_{disease.lower()}.csv"
        csv_path = os.path.join(self.base_path, disease, csv_name)
        return pd.read_csv(csv_path)

    def load_patient(self, disease, patient_row):
        """
        Loads images for a single patient record (row from the CSV).
        """
        # 1. Identify the split (train/test/val) to find the right subfolder
        # Note: Your folders on the cluster are named 'Training', 'Test', 'Validation'
        split_map = {
            'training': 'Training',
            'test': 'Test',
            'validation': 'Validation'
        }
        split_folder = split_map.get(patient_row['use'], patient_row['use'])
        
        # 2. Construct paths
        npz_filename = patient_row['filename']  # e.g., data_05995.npz
        patient_id = npz_filename.replace('data_', '').replace('.npz', '')
        jpg_filename = f"slo_fundus_{patient_id}.jpg"
        
        disease_dir = os.path.join(self.base_path, disease, split_folder)
        
        # 3. Load NPZ (OCT/Tensors)
        npz_path = os.path.join(disease_dir, npz_filename)
        container = np.load(npz_path)
        oct_data = container['oct_bscans']
        
        # 4. Load JPG (Fundus Photo)
        jpg_path = os.path.join(disease_dir, jpg_filename)
        fundus_img = Image.open(jpg_path)
        
        return {
            "metadata": patient_row.to_dict(),
            "oct_tensors": oct_data,
            "fundus_img": fundus_img
        }

#loader = GenericEyeLoader("/lustre/fs1/home/yu395012/OphthalmicAgent/")
#
#disease_name = 'Glaucoma'
#df_glaucoma = loader.get_metadata(disease_name)
#
## Get only the test patients
#test_patients = df_glaucoma[df_glaucoma['use'] == 'test']
#
#if not test_patients.empty:
#    # 1. Load the first test patient
#    patient_data = loader.load_patient(disease_name, test_patients.iloc[0])
#    
#    # 2. Print the whole metadata
#    print("\n--- PATIENT METADATA ---")
#    pprint(patient_data['metadata'])
#    
#    # 3. Visualize the Fundus Image
#    print("\n--- VISUALIZING FUNDUS IMAGE ---")
#    fundus_img = patient_data['fundus_image']
#    
#    # Save the plot so you can view it on the cluster
#    output_filename = "fundus_check.png"
#    plt.savefig(output_filename)
#    print(f"Visualization saved as {output_filename}")
#    
##    print(f"OCT Shape: {patient_data['oct_tensors'][list(patient_data['oct_tensors'].keys())[0]].shape}")
#    
#    # See what's actually inside
##    print(f"Inside this NPZ, I found these keys: {patient_data['oct_tensors'].files}")
#    
#    print(f"Shape of OCT: {patient_data['oct_tensors'].shape}")       
#else:
#    print("No test patients found in the CSV.")
##
##volume = patient_data['oct_tensors']
##
##single_slice = volume[100, :, :] 
##print(single_slice.shape) # Result: (200, 200)
