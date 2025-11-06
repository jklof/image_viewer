import logging
from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)


class ImageEmbedder:
    """
    An embedder that uses a configurable CLIP model for state-of-the-art
    search performance. It dynamically determines embedding size from the
    loaded model. Can be configured to run on CPU only.
    """

    def __init__(self, model_id: str, use_cpu_only: bool = False):
        if use_cpu_only:
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"ImageEmbedder initializing. Using device: {self.device}")

        # Eagerly load the model.
        # Use torch.float16 for GPU to save VRAM and improve performance.
        # Use default precision (float32) for CPU as float16 can be slower.
        model_kwargs = {}
        if self.device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16

        logger.info(f"Loading model '{model_id}'...")
        self.model = CLIPModel.from_pretrained(model_id, **model_kwargs).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        logger.info("Model loaded successfully.")

        # --- Dynamically determine and store model properties ---
        self.model_id = model_id
        # The embedding dimension is found in the model's config.
        embedding_dim = self.model.config.projection_dim
        self.embedding_shape = (1, embedding_dim)
        self.embedding_dtype = np.float32

        logger.info(f"Model '{model_id}' loaded with embedding dimension {embedding_dim}.")

    @torch.no_grad()
    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Generates a normalized embedding for a single PIL Image."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        features = self.model.get_image_features(**inputs)
        features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(self.embedding_dtype)

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        """Generates a normalized embedding for a single string of text."""
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        features = self.model.get_text_features(**inputs)
        features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(self.embedding_dtype)

    @torch.no_grad()
    def embed_batch(self, images: List[Image.Image], batch_size: int = 4) -> List[np.ndarray]:
        """Generates embeddings for a list of PIL Images in batches."""
        all_embeddings = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt", padding=True).to(self.device)
            features = self.model.get_image_features(**inputs)
            features /= features.norm(dim=-1, keepdim=True)
            all_embeddings.extend([emb.cpu().numpy().astype(self.embedding_dtype) for emb in features])
        return all_embeddings
