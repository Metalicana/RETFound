#import pandas as pd
#import numpy as np
#import os
#from PIL import Image
#import matplotlib.pyplot as plt
#from pprint import pprint
#
#class GenericEyeLoader:
#    def __init__(self, base_path): 
#        """
#        base_path: The root folder containing DR, AMD, and Glaucoma folders.
#        """
#        self.base_path = base_path
#
#    def get_metadata(self, disease):
#        """Loads the correct CSV based on disease name."""
#        # Standardizing naming: e.g., 'glaucoma' -> 'data_summary_glaucoma.csv'
#        csv_name = f"data_summary_{disease.lower()}.csv"
#        csv_path = os.path.join(self.base_path, disease, csv_name)
#        return pd.read_csv(csv_path)
#
#    def load_patient(self, disease, patient_row):
#        """
#        Loads images for a single patient record (row from the CSV).
#        """
#        # 1. Identify the split (train/test/val) to find the right subfolder
#        # Note: Your folders on the cluster are named 'Training', 'Test', 'Validation'
#        split_map = {
#            'training': 'Training',
#            'test': 'Test',
#            'validation': 'Validation'
#        }
#        split_folder = split_map.get(patient_row['use'], patient_row['use'])
#        
#        # 2. Construct paths
#        npz_filename = patient_row['filename']  # e.g., data_05995.npz
#        patient_id = npz_filename.replace('data_', '').replace('.npz', '')
#        jpg_filename = f"slo_fundus_{patient_id}.jpg"
#        
#        disease_dir = os.path.join(self.base_path, disease, split_folder)
#        
#        # 3. Load NPZ (OCT/Tensors)
#        npz_path = os.path.join(disease_dir, npz_filename)
#        container = np.load(npz_path)
#        
#        if(disease == "AMD"): 
#           
#          amd_map = {
#              'not.in.icd.table': 0., 'no.amd.diagnosis': 0.,
#              'early.dry': 1., 'intermediate.dry': 2., 
#              'advanced.atrophic.dry.with.subfoveal.involvement': 3.,
#              'advanced.atrophic.dry.without.subfoveal.involvement': 3.,
#              'wet.amd.active.choroidal.neovascularization': 3.,
#              'wet.amd.inactive.choroidal.neovascularization': 3.,
#              'wet.amd.inactive.scar': 3.
#          }
#          
#          stage = amd_map.get(container['amd_condition'].item(), "Unknown")
#          
#          print(f"\nDisease folder is {disease} and the ground truth is {container['amd_condition']} which is stage {stage}")
#          
#        elif(disease == "DR"): 
#        
#          dr_map = {
#              'not.in.icd.table': 0., 'no.dr.diagnosis': 0.,
#              'mild.npdr': 0., 'moderate.npdr': 0.,
#              'severe.npdr': 1., 'pdr': 1.
#          }
#          
#          stage = dr_map.get(container['dr_subtype'].item(), "Unknown")
#          print(f"\nDisease folder is {disease} and the ground truth is {stage} (0 means negative, 1 means positive)")
#          
#        elif(disease == "Glaucoma"): 
#          stage = container['glaucoma']
#          print(f"\nDisease folder is {disease} and the ground truth is {container['glaucoma']} (0 means negative, 1 means positive)")
#        
#        oct_volume = container['oct_bscans']  
#        
#        # Logic for extracting the center slice
#        mid_idx = oct_volume.shape[0] // 2
#        
#        oct_slice = oct_volume[mid_idx]
#        # Intensity Normalization
#        if oct_slice.max() <= 1.0:
#            oct_slice = (oct_slice * 255).astype(np.uint8)
#        else:
#            oct_slice = oct_slice.astype(np.uint8)
#        oct_image = oct_slice.astype(np.uint8)       
#  
#        fundus_img = container['slo_fundus']
#        fundus_img = fundus_img.astype(np.float32)
#        f_min, f_max = fundus_img.min(), fundus_img.max()
#        if f_max - f_min > 0:
#          fundus_img = 255 * (fundus_img - f_min) / (f_max - f_min)
#        fundus_img = fundus_img.astype(np.uint8)
#        
#        return {
#            "metadata": patient_row.to_dict(),
#            "oct_tensors": oct_volume,
#            "fundus_img": fundus_img,
#            "directory": npz_path,
#            "stage": stage,
#            "oct_img": oct_image
#        }

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from PIL import Image
from pprint import pprint

class ExcelEyeLoader:
    def __init__(self, excel_path):
        """
        excel_path: Path to the new master Excel sheet containing file paths and ground truth columns.
        """
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Master Excel file not found at: {excel_path}")
            
        # Supports both Excel formats (.xlsx / .xls) and CSVs if you change extension
        if excel_path.endswith('.csv'):
            self.df = pd.read_csv(excel_path)
        else:
            self.df = pd.read_excel(excel_path)
            
        print(f"Successfully loaded Excel sheet. Total records available: {len(self.df)}")

    def get_records_by_disease(self, disease, limit=250):
        """
        Filters the Excel master sheet for a specific disease and returns the top N records.
        Assumes your excel sheet contains a column identifying the disease task (e.g., 'Task_Folder' or 'disease_folder').
        """
        # Automatically handles case variations
        disease_col = [col for col in self.df.columns if col.lower() in ['task_folder', 'disease_folder', 'disease']]
        if not disease_col:
            raise KeyError("Could not find a disease folder column in your Excel file (expected 'Task_Folder' or similar).")
            
        disease_mask = self.df[disease_col[0]].str.lower() == disease.lower()
        filtered_df = self.df[disease_mask].head(limit)
        
        print(f"Extracted {len(filtered_df)} sample paths for {disease} task.")
        return filtered_df

    def load_patient_from_excel_row(self, patient_row):
        """
        Loads images and structures directly utilizing columns mapped inside the excel sheet.
        """
        # 1. Dynamically extract structural file path column from Excel
        path_col = [col for col in patient_row.index if col.lower() in ['filepath', 'file_path', 'path', 'npz_path', 'filename']]
        if not path_col:
            raise KeyError("Could not find a column mapping image file paths in your Excel file.")
        
        npz_path = patient_row[path_col[0]]
        
        # Determine current task/disease context
        disease_col = [col for col in patient_row.index if col.lower() in ['task_folder', 'disease_folder', 'disease']][0]
        current_disease = str(patient_row[disease_col]).strip()

        # 2. Extract Ground Truth directly from the Excel column
        gt_col = [col for col in patient_row.index if col.lower() in ['ground_truth', 'groundtruth', 'gt', 'label']]
        if not gt_col:
            raise KeyError("Could not find a 'Ground_Truth' column in your Excel file.")
            
        raw_gt = patient_row[gt_col[0]]

        # 3. Apply Staging/Binarization mapping directly from Excel input values
        if current_disease.lower() == "amd":
            # BINARY CLASSIFICATION LOGIC: 0 is negative/healthy, any value greater than 0 is positive/disease
            if float(raw_gt) > 0:
                stage = 1.0
            else:
                stage = 0.0
            print(f"\nDisease: AMD | Excel Original GT: {raw_gt} -> Converted to Binary: {stage} (0: Negative, 1: Positive)")
            
        elif current_disease.lower() == "dr":
            stage = float(raw_gt)
            print(f"\nDisease: DR | Excel Ground Truth: {stage} (0: Negative, 1: Positive)")
            
        else: # Glaucoma / Default
            stage = float(raw_gt)
            print(f"\nDisease: {current_disease} | Excel Ground Truth: {stage} (0: Negative, 1: Positive)")

        # 4. Load NPZ (OCT Arrays / Tensors) directly from destination path
        container = np.load(npz_path)
        oct_volume = container['oct_bscans']  
        
        # Extract middle structural frame slice
        mid_idx = oct_volume.shape[0] // 2
        oct_slice = oct_volume[mid_idx]
        
        num_slices = 8
        
        indices = np.linspace(
            0,
            oct_volume.shape[0] - 1,
            num_slices,
            dtype=int
        )

        selected_slices = oct_volume[indices]
        
#        images = []
#
#        for oct_slice in selected_slices:
#            if oct_slice.max() <= 1.0:
#                oct_slice = (oct_slice * 255).astype(np.uint8)
#            else:
#                oct_slice = oct_slice.astype(np.uint8)
#        
#            img = Image.fromarray(oct_slice).convert("RGB")
#        
#            images.append(img)
#        
#        oct_image = torch.stack(images)   # (16,3,224,224)
#            
#        # Contrast Intensity Normalization
#        if oct_slice.max() <= 1.0:
#            oct_slice = (oct_slice * 255).astype(np.uint8)
#        else:
#            oct_slice = oct_slice.astype(np.uint8)
#        oct_image = oct_slice.astype(np.uint8)       
  
        fundus_img = container['slo_fundus']
        fundus_img = fundus_img.astype(np.float32)
        f_min, f_max = fundus_img.min(), fundus_img.max()
        if f_max - f_min > 0:
            fundus_img = 255 * (fundus_img - f_min) / (f_max - f_min)
        fundus_img = fundus_img.astype(np.uint8)
        
        # --- VERIFICATION PRINT BLOCK ---
        print(f"\n{'='*15} DATA BATCH VERIFICATION {'='*15}")
        print(f"File Source Directory : {npz_path}")
        print(f"Mapped Clinical Stage : {stage}")
        
        # Print data dimensions/shapes for image validation
        print(f"OCT Volume (Tensors)  : Shape {oct_volume.shape} (Slices, Height, Width)")
        print(f"SLO Fundus Image      : Shape {fundus_img.shape} (Height, Width)")
        print(f"{'='*55}\n")
        
        return {
            "metadata": patient_row.to_dict(),
            "oct_tensors": oct_volume,
            "fundus_img": fundus_img,
            "directory": npz_path,
            "stage": stage,
            "oct_img": selected_slices,
            "middle_oct": oct_slice
        }