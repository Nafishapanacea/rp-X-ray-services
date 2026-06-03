import torch
from torchvision import transforms as T
from PIL import Image
import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
from transformers import AutoProcessor
from src.configuration.config import IMAGE_SIZE, MEAN, STD, DEVICE, MODEL_NAME, D_Type

processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

preprocess = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])


def prepare_image(image: Image.Image):
    img = image.convert("RGB")
    tensor = preprocess(img).unsqueeze(0)  # [1,3,512,512]
    return tensor.to(DEVICE)


def disease_classify(model, image_tensor, LABELS):
    with torch.no_grad():
        logits, pooling_attn_weights = model(image_tensor)
        probs = torch.sigmoid(logits)

    probs = probs.cpu().numpy()[0]
    selected_labels = [label for label, prob in zip(LABELS, probs) if prob > 0.5]
    
    print('probs', probs)
    result = {label: float(prob) for label, prob in zip(LABELS, probs)}
    print("Predictions:", result)
    # return selected_labels
    return {
        "probs": result, 
        "labels": selected_labels,
        "attn_weights": pooling_attn_weights
    }


# TB prediction
def tb_predict(model, image_tensor):
    with torch.no_grad():
        logits, attention, pooling_attn_weights = model(image_tensor)
    prob    = torch.sigmoid(logits).squeeze().item()   # scalar float
    pred    = 1 if prob >= 0.5 else 0
    finding = "TB positive" if pred == 1 else "TB Negative"

    return finding, pooling_attn_weights

# Normal-abnormal prediction
def na_predict(model, image_tensor, view_tensor, sex_tensor):
    with torch.no_grad():
        logits, attention, pooling_attn_weights = model(image_tensor, view_tensor, sex_tensor)
    prob    = torch.sigmoid(logits).squeeze().item()   # scalar float
    pred    = 1 if prob >= 0.5 else 0
    finding = "Abnormal" if pred == 1 else "Normal"

    return finding, pooling_attn_weights

def preprocess_image(image_path: str) -> torch.Tensor:

    image = Image.open(image_path)

    # ── Convert any mode (L, RGBA, P ...) to RGB ─────────────────────────
    image = image.convert("RGB")

    # ── XraySigLIP processor ─────────────────────────────────────────────
    inputs       = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]            # (1, C, H, W)
    pixel_values = pixel_values.to(DEVICE, D_Type)

    return pixel_values

def extract_dicom_metadata(dicom_path):
    ds = pydicom.dcmread(dicom_path)

    # Safe extraction with fallback
    sex = getattr(ds, "PatientSex", None)
    view = getattr(ds, "ViewPosition", None)

    return sex, view

def dicom_to_image(dicom_path,output_path,format="png"):
    try:
        dicom=pydicom.dcmread(dicom_path)
        pixel_array = dicom.pixel_array

        pixel_array=(pixel_array-pixel_array.min())/(pixel_array.max()-pixel_array.min())*255
        pixel_array=pixel_array.astype(np.uint8)

        if dicom.PhotometricInterpretation=="MONOCHROME1":
            pixel_array=255-pixel_array

        if format=="jpg":
            cv2.imwrite(output_path,pixel_array)
        else:
            Image.fromarray(pixel_array).save(output_path)

    except Exception as e:
        raise RuntimeError(f"Failed to convert Dicom to {format}:{str(e)}")


# def register_pooling_hook(model):
#     attn_store = {}

#     def pool_hook(module, input, output):
#         attn_store["weights"] = output[1]  # [B, 1, 1024]

#     model.vision_encoder.head.attention.register_forward_hook(pool_hook)
#     return attn_store


def generate_heatmap(pixel_values, attn_weights, img_size=512):
    """
    Returns a numpy image for Gradio
    """

    attn = attn_weights[0, 0]        # [N]
    attn = attn / (attn.max() + 1e-8)

    heatmap = attn.reshape(32, 32)

    heatmap_up = F.interpolate(
        heatmap.unsqueeze(0).unsqueeze(0),
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False
    )[0, 0]

    heatmap_np = heatmap_up.cpu().numpy()
    heatmap_np = (heatmap_np * 255).astype(np.uint8)

    heatmap_color = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)

    img = pixel_values[0].permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())
    img = (img * 255).astype(np.uint8)

    overlay = cv2.addWeighted(img, 0.75, heatmap_color, 0.25, 0)

    return overlay 
