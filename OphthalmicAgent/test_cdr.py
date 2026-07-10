import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from scipy.ndimage import find_objects

def preprocess_slo_image(image_path):
    """
    Loads an SLO image, converts it to RGB (duplicating channels if grayscale),
    and applies histogram equalization to help match fundus image intensity profiles.
    """
    img = Image.open(image_path)
    
    # Force convert to RGB because the model expects a 3-channel input
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    # Optional: Equalize contrast to enhance cup/disc boundaries in monochromatic SLO
    img = ImageOps.equalize(img)
    return img

def segment_cup_and_disc(image_path):
    # 1. Load the pre-trained SegFormer model and image processor
    model_name = "pamixsun/segformer_for_optic_disc_cup_segmentation"
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # 2. Load and pre-process the SLO image
    raw_image = preprocess_slo_image(image_path)
    
    # 3. Run inference
    inputs = processor(images=raw_image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
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
    
    # 6. Visualization
    visualize_results(raw_image, prediction)
    
    return prediction

def visualize_results(original_image, mask):
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
    plt.savefig("cdr_output.png")
 


def calculate_cdr_from_mask(prediction_mask):
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
        print("Error: Could not find both optic disc and cup in the mask.")
        return None, None

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
    
    return v_cdr, h_cdr

 

# --- Run the pipeline ---
if __name__ == "__main__":
    # Replace with the path to your SLO image file
    image_file_path = "clahe.png" 
    
    try:
        prediction = segment_cup_and_disc(image_file_path)
        
        v_cdr, h_cdr = calculate_cdr_from_mask(prediction)  
        
        if v_cdr is not None:
          print("--- CDR Analysis Results ---")
          print(f"Vertical CDR (Clinical Focus): {v_cdr:.3f}")
          print(f"Horizontal CDR:               {h_cdr:.3f}")
          
          # Brief clinical context output
          if v_cdr > 0.7:
              print("Note: High vertical CDR detected (> 0.7). Potential glaucomatous risk.")
          elif v_cdr > 0.4:
              print("Note: Moderate vertical CDR detected (0.4 - 0.7). Check for asymmetry.")
          else:
              print("Note: Normal vertical CDR detected (< 0.4).") 
              
    except FileNotFoundError:
        print(f"Please provide a valid image path. '{image_file_path}' was not found.")