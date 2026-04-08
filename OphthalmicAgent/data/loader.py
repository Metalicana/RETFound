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
        
#        print(f"path is: {npz_path}")
        
        if(disease == "AMD"): 
           
          amd_map = {
              'not.in.icd.table': 0., 'no.amd.diagnosis': 0.,
              'early.dry': 1., 'intermediate.dry': 2., 
              'advanced.atrophic.dry.with.subfoveal.involvement': 3.,
              'advanced.atrophic.dry.without.subfoveal.involvement': 3.,
              'wet.amd.active.choroidal.neovascularization': 3.,
              'wet.amd.inactive.choroidal.neovascularization': 3.,
              'wet.amd.inactive.scar': 3.
          }
          
          stage = amd_map.get(container['amd_condition'].item(), "Unknown")
          
          print(f"\nDisease folder is {disease} and the ground truth is {container['amd_condition']} which is stage {stage}")
          
        elif(disease == "DR"): 
        
          dr_map = {
              'not.in.icd.table': 0., 'no.dr.diagnosis': 0.,
              'mild.npdr': 0., 'moderate.npdr': 0.,
              'severe.npdr': 1., 'pdr': 1.
          }
          
          stage = dr_map.get(container['dr_subtype'].item(), "Unknown")
          print(f"\nDisease folder is {disease} and the ground truth is {stage} (0 means negative, 1 means positive)")
          
        elif(disease == "Glaucoma"): 
          stage = container['glaucoma']
          print(f"\nDisease folder is {disease} and the ground truth is {container['glaucoma']} (0 means negative, 1 means positive)")
        
        oct_volume = container['oct_bscans']  
        
        # Logic for extracting the center slice
        mid_idx = oct_volume.shape[0] // 2
        
        oct_slice = oct_volume[mid_idx]
        # Intensity Normalization
        if oct_slice.max() <= 1.0:
            oct_slice = (oct_slice * 255).astype(np.uint8)
        else:
            oct_slice = oct_slice.astype(np.uint8)
        oct_image = oct_slice.astype(np.uint8)       
  
        fundus_img = container['slo_fundus']
        fundus_img = fundus_img.astype(np.float32)
        f_min, f_max = fundus_img.min(), fundus_img.max()
        if f_max - f_min > 0:
          fundus_img = 255 * (fundus_img - f_min) / (f_max - f_min)
        fundus_img = fundus_img.astype(np.uint8)
        
        return {
            "metadata": patient_row.to_dict(),
            "oct_tensors": oct_volume,
            "fundus_img": fundus_img,
            "directory": npz_path,
            "stage": stage,
            "oct_img": oct_image
        }