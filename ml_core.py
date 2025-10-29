# ml_core.py

import logging
from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)

# The best single model that fits comfortably in 8GB VRAM.
MODEL_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

# The embedding dimension for ViT-H is 1024.
EMBEDDING_DTYPE = np.float32
EMBEDDING_SHAPE = (1, 1024)


class ImageEmbedder:
    """
    An embedder that uses a single, powerful CLIP model (ViT-H-14) for
    state-of-the-art search performance with excellent efficiency.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Eagerly load the model for best performance.
        # Use torch_dtype=torch.float16 to ensure it loads in half-precision.
        logger.info(f"Loading model '{MODEL_ID}'...")
        self.model = CLIPModel.from_pretrained(MODEL_ID, dtype=torch.float16).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(MODEL_ID)
        logger.info("Model loaded successfully.")

    @torch.no_grad()
    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Generates a normalized embedding for a single PIL Image."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        features = self.model.get_image_features(**inputs)
        features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(EMBEDDING_DTYPE)

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        """Generates a normalized embedding for a single string of text."""
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        features = self.model.get_text_features(**inputs)
        features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(EMBEDDING_DTYPE)

    @torch.no_grad()
    def embed_batch(self, images: List[Image.Image], batch_size: int = 4) -> List[np.ndarray]:
        """Generates embeddings for a list of PIL Images in batches."""
        all_embeddings = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt", padding=True).to(self.device)
            features = self.model.get_image_features(**inputs)
            features /= features.norm(dim=-1, keepdim=True)
            all_embeddings.extend([emb.cpu().numpy().astype(EMBEDDING_DTYPE) for emb in features])
        return all_embeddings