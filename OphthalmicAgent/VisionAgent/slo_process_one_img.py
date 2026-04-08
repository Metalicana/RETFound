import sys
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

MIRAGE_DIR = os.path.abspath("./VisionAgent/MIRAGE")
sys.path.append(MIRAGE_DIR)

from linear_probing_slo import get_model_slo

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_single_image(model, image_path, device):
    
    data = np.load(image_path)
    fundus_img = data['slo_fundus']
    if fundus_img.max() <= 1.0:
        fundus_img = (fundus_img * 255).astype(np.uint8)
    else:
        fundus_img = fundus_img.astype(np.uint8)
        
    # Convert to RGB (RETFound expects 3 channels)
    image = Image.fromarray(fundus_img).convert('L')   
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
     
    input_tensor = transform(image).unsqueeze(0).to(device) 

    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        
        amd_probs = torch.sigmoid(outputs['amd']).cpu().numpy()[0]
        dr_prob = torch.sigmoid(outputs['dr']).cpu().numpy()[0][0]
        glaucoma_prob = torch.sigmoid(outputs['glaucoma']).cpu().numpy()[0][0]

    # 4. Display Results
    print(f"\nResults for: {os.path.basename(image_path)}")
    print("-" * 30)
    print(f"AMD Stage >= 1 Prob: {amd_probs[0]:.4f}")
    print(f"AMD Stage >= 2 Prob: {amd_probs[1]:.4f}")
    print(f"AMD Stage >= 3 Prob: {amd_probs[2]:.4f}")
    print(f"DR Probability:      {dr_prob:.4f}")
    print(f"Glaucoma Prob:       {glaucoma_prob:.4f}")
    
    # Simple Threshold Logic
    if dr_prob > 0.5: print("Prediction: Diabetic Retinopathy detected")
    if glaucoma_prob > 0.5: print("Prediction: Glaucoma detected")

    return {'amd': amd_probs, 'dr': dr_prob, 'glaucoma': glaucoma_prob}

if __name__ == "__main__":
    # Setup Model
    original_dir = os.getcwd()
    os.chdir(MIRAGE_DIR)
    model = get_model().to(DEVICE) # Now it finds the weights in the current folder
    os.chdir(original_dir)
    
    model.load_state_dict(torch.load("slo_model_best.pth", map_location=DEVICE))

    # Path to a specific SLO image (.png, .jpg, etc.)
    # If your data is in .npz, you'll need to extract the 'slo_fundus' array first
    test_image_path = "/lustre/fs1/home/yu395012/RETFound/OphthalmicAgent/data/AMD/Test/data_07007.npz"

    if os.path.exists(test_image_path):
        predict_single_image(model, test_image_path, DEVICE)
    else:
        print("Please provide a valid image path.")