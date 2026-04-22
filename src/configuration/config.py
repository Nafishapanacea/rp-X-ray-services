import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DICOM_TEMP_PATH = '/tmp/dicom_uploads'
outputDir = 'outputs'

D_Type  = torch.float32
MODEL_NAME = "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli"
NA_CHECKPOINT_PATH = "/home/ubuntu/Documents/rp-x-ray-normal-abnormal/rp-normal-abnormal-service/version3.pt"
TB_CHECKPOINT_PATH = "/home/ubuntu/Documents/tb-classification-service/best_model_attention_loss.pth"
DISEASE_CLASSIFICATION_CHECKPOINT_PATH = '/home/ubuntu/Documents/rp-X-ray-services/stage3_best.pt'

VIEW_MAP = {"AP": 1, "PA": 0, "Lateral": 2}
SEX_MAP  = {"M": 0, "F": 1}

IMAGE_SIZE = 512
NUM_LABELS = 18

# same normalization as training
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD  = [0.26862954, 0.26130258, 0.27577711]

LABELS = ["Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia", "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia", "Lung Lesion", "Fracture", "Lung Opacity", "Enlarged Cardiomediastinum"]