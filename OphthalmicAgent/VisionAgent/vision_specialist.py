import os
import torch
import numpy as np
import cv2
import torch.nn.functional as F

# Absolute imports from your package structure
from data.loader import GenericEyeLoader
from VisionAgent.models_vit import vit_large_patch16

class VisionSpecialist:
    def __init__(self, weight_path, device=None):
        """
        Initializes the RETFound backbone for feature extraction.
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Initializing Vision Specialist on: {self.device}")

        # 1. Load Architecture (ViT-Base)
        self.model = vit_large_patch16(
            num_classes=0,  # 0 for feature extraction (removes the head)
            drop_path_rate=0.1,
            global_pool=''
        )

        # 2. Load Pre-trained Weights
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weights not found at {weight_path}")
            
        checkpoint = torch.load(weight_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.model.to(self.device)
        self.model.eval()
        print("RETFound weights loaded successfully.")

    def _preprocess(self, oct_volume):
        """
        Converts (200, 200, 200) OCT to (1, 3, 224, 224) torch tensor.
        """
        # Pick center slice (index 100) - common practice for foveal analysis
        slice_2d = oct_volume[100, :, :]

        # Normalize to [0, 1]
        slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8)

        # Resize to model input size (224x224)
        resized = cv2.resize(slice_2d, (224, 224), interpolation=cv2.INTER_LINEAR)

        # Convert to Tensor and add Channel/Batch dims: [1, 1, 224, 224]
        tensor = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0)

        # Repeat to 3 channels (RGB-like) for the ViT: [1, 3, 224, 224]
        tensor = tensor.repeat(1, 3, 1, 1)
        
        return tensor.to(self.device)

    def get_features(self, oct_volume):
        """
        Passes the processed OCT through RETFound and returns the 768-dim vector.
        """
        input_tensor = self._preprocess(oct_volume)
        
        with torch.no_grad():
            features = self.model(input_tensor)
            
        return features.cpu().numpy().flatten()

## --- Integration Test Logic ---
#if __name__ == "__main__":
#    # 1. Setup paths
#    BASE_PATH = "/lustre/fs1/home/yu395012/OphthalmicAgent/"
#    WEIGHTS = os.path.join(BASE_PATH, "weights/RETFound_mae_natureOCT.pth")
#    
#    # 2. Initialize Agent
#    vision_agent = VisionSpecialist(WEIGHTS)
#    
#    disease = 'DR'
#    
#    # 3. Load a sample using your Generic Loader
#    loader = GenericEyeLoader(BASE_PATH)
#    df = loader.get_metadata(disease)
#    
#    # Grab the first test patient
#    test_rows = df[df['use'] == 'test']
#    if not test_rows.empty:
#        patient_record = loader.load_patient(disease, test_rows.iloc[1])
#        oct_data = patient_record['oct_tensors']
#        
#        # 4. Extract Features
#        features = vision_agent.get_features(oct_data)
#        
#        print(f"\nProcessing File: {patient_record['metadata']['filename']}")
#        print(f"Feature Vector Shape: {features.shape}")
#        print(f"First 5 features: {features[:5]}")
#    else:
#        print("No test data found to process.")