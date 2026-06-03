import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from src.configuration.config import DEVICE, MODEL_NAME, NUM_LABELS, DISEASE_CLASSIFICATION_CHECKPOINT_PATH


class CXRMultiLabel(nn.Module):
    def __init__(self, vision_encoder, num_labels):
        super().__init__()
        self.vision_encoder = vision_encoder
        in_dim = vision_encoder.config.hidden_size

        self.vision_encoder.head.attention.register_forward_hook(
            self._pooling_attn_hook
        )

        self.head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels)
        )

    def _pooling_attn_hook(self, module, input, output):
        self.pooling_attn_weights = output[1]  

    def forward(self, pixel_values):
        out = self.vision_encoder(pixel_values=pixel_values, return_dict=True)
        cls = out.last_hidden_state[:, 0, :]
        logits = self.head(cls)
        return logits, self.pooling_attn_weights


def load_disease_classification_model():
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)

    vision_full = AutoModel.from_pretrained(
        MODEL_NAME,
        config=config,
        trust_remote_code=True
    )

    vision = vision_full.vision_model.to(DEVICE)

    model = CXRMultiLabel(vision, NUM_LABELS).to(DEVICE)

    # ckpt = torch.load(DISEASE_CLASSIFICATION_CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    # model.load_state_dict(ckpt["model_state"])

    model.eval()
    return model