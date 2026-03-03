# --- Inference and Submission Generation ---
import torch
import pandas as pd
from tqdm.auto import tqdm

def run_inference(model, loader, le, threshold, device):
    model.eval()
    results = []
    
    with torch.no_grad():
        for images, ids in tqdm(loader, desc="Inference"):
            images = images.to(device)
            outputs = model(images)
            
            # Convert outputs to probabilities
            probs = torch.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, dim=1)
            
            for i in range(len(ids)):
                # If the model is not confident enough, it's an unknown writer (-1)
                if confidences[i].item() < threshold:
                    writer_id = "-1"
                else:
                    # Convert the numerical class back to the original W01, W02...
                    writer_id = le.inverse_transform([preds[i].cpu().item()])[0]
                
                results.append({
                    "image_id": ids[i].item(),
                    "writer_id": writer_id
                })
                
    return pd.DataFrame(results)
