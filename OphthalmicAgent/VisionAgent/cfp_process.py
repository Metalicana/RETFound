#processing the whole test data for AMD without 0,1,2,3

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# Import your existing classes from your training file
from VisionAgent.linear_probing_fundus import FairVisionNPZ, get_model 

DATA_ROOT = "/lustre/fs1/home/yu395012/RETFound/OphthalmicAgent/data/"
MODEL_WEIGHTS = "fundus_model.pth"
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_trained_model(weights_path):
    print(f"Rebuilding model architecture and loading weights from {weights_path}...")
    # This calls your get_model function which has the frozen backbone
    model = get_model()
    
    # Load the state_dict saved during training
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=False))
    model.to(DEVICE)
    model.eval() # Set to evaluation mode
    return model

def main():
    # 1. Setup Data
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_ds = FairVisionNPZ(DATA_ROOT, split='Test', transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=8)

    # 2. Load Model
    model = load_trained_model(MODEL_WEIGHTS)

    # 3. Run Inference
    results = []
    print(f"Running Fundus Specialist Inference on {len(test_ds)} cases...")

    with torch.no_grad():
        for i, (imgs, labels, races) in enumerate(tqdm(test_loader)):
            imgs = imgs.to(DEVICE)
            
            # Forward pass
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            # Metadata for the CSV
            for j in range(len(imgs)):
                # Get the original file path from the dataset
                file_info = test_ds.files[i * 32 + j] 
                
                results.append({
                    'file_path': file_info['path'],
                    'source_folder': file_info['source'],
                    'prob_AMD': probs[j, 0],
                    'prob_DR': probs[j, 1],
                    'prob_Glaucoma': probs[j, 2],
                    'target_AMD': labels[j, 0].item(),
                    'target_DR': labels[j, 1].item(),
                    'target_Glaucoma': labels[j, 2].item()
                })

    # 4. Save to CSV for the 11-Agent System
    df = pd.DataFrame(results)
    df.to_csv("fundus_specialist_inference.csv", index=False)
    print("\nInference Complete! Results saved to 'fundus_specialist_inference.csv'")

    # 5. Quick Performance Check (AUC)
    print("\n--- Test Set Performance ---")
    for idx, disease in enumerate(['AMD', 'DR', 'Glaucoma']):
        y_true = df[f'target_{disease}'].values
        y_score = df[f'prob_{disease}'].values
        
        # Only calculate if there are positive examples in the test set
        if len(np.unique(y_true[y_true != -1])) > 1:
            auc = roc_auc_score(y_true[y_true != -1], y_score[y_true != -1])
            print(f"{disease} AUC: {auc:.4f}")

if __name__ == "__main__":
    main()
