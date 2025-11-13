import logging
import numpy as np
from PIL import Image

# All heavy imports are now inside methods/constructors.

logger = logging.getLogger(__name__)


class ImageEmbedder:
    """
    An embedder that uses a configurable CLIP model. JIT imports ensure
    that heavy libraries are only loaded when an instance is created.
    """

    def __init__(self, model_id: str, use_cpu_only: bool = False):
        # --- Heavy imports happen here, only when an instance is needed ---
        logger.info("Importing ML libraries...")
        import torch
        from transformers import CLIPModel, CLIPProcessor

        if use_cpu_only:
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"ImageEmbedder initializing. Using device: {self.device}")

        model_kwargs = {}
        if self.device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16

        logger.info(f"Loading model '{model_id}'...")
        self.model = CLIPModel.from_pretrained(model_id, **model_kwargs).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        logger.info("Model loaded successfully.")

        self.model_id = model_id
        embedding_dim = self.model.config.projection_dim
        self.embedding_shape = (1, embedding_dim)
        self.embedding_dtype = np.float32
        logger.info(f"Model '{model_id}' loaded with embedding dimension {embedding_dim}.")

    def embed_image(self, image: Image.Image) -> np.ndarray:
        import torch

        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            features = self.model.get_image_features(**inputs)
            features /= features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy().astype(self.embedding_dtype)

    def embed_text(self, text: str) -> np.ndarray:
        import torch

        with torch.no_grad():
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)
            features = self.model.get_text_features(**inputs)
            features /= features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy().astype(self.embedding_dtype)

    def embed_batch(self, images: list[Image.Image], batch_size: int = 4) -> list[np.ndarray]:
        import torch

        with torch.no_grad():
            all_embeddings = []
            for i in range(0, len(images), batch_size):
                batch = images[i : i + batch_size]
                inputs = self.processor(images=batch, return_tensors="pt", padding=True).to(self.device)
                features = self.model.get_image_features(**inputs)
                features /= features.norm(dim=-1, keepdim=True)
                all_embeddings.extend([emb.cpu().numpy().astype(self.embedding_dtype) for emb in features])
            return all_embeddings
