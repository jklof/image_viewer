import logging
import threading
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ImageEmbedder:
    """
    An embedder that uses a configurable CLIP model.
    """

    def __init__(self, model_id: str, use_cpu_only: bool = False, idle_timeout: float = 30.0):
        self.idle_timeout = idle_timeout
        self.lock = threading.Lock()
        self.timer = None
        self.is_offloaded = True

        self.model = None
        self.processor = None
        self.compute_device = None
        self.model_id = None
        self.embedding_shape = None
        self.embedding_dtype = np.float32

        logger.info("Importing ML libraries...")
        import torch
        from transformers import CLIPModel, CLIPProcessor

        if use_cpu_only:
            self.compute_device = "cpu"
        elif torch.cuda.is_available():
            self.compute_device = "cuda"
        # MPS has known precision bugs with CLIP layernorm that severely degrade 
        # embedding quality and similarity search order. Falling back to CPU.
        else:
            self.compute_device = "cpu"

        logger.info(f"ImageEmbedder initializing. Compute device: {self.compute_device}")

        model_kwargs = {}
        if self.compute_device == "cuda":
            model_kwargs["dtype"] = torch.float16

        logger.info(f"Loading model '{model_id}' to RAM (CPU)...")
        # Load initially to CPU to save VRAM
        self.model = CLIPModel.from_pretrained(model_id, **model_kwargs).to("cpu")
        self.processor = CLIPProcessor.from_pretrained(model_id)
        logger.info("Model loaded successfully.")

        self.model_id = model_id
        embedding_dim = self.model.config.projection_dim
        self.embedding_shape = (1, embedding_dim)

    def _reset_timer(self):
        """Restarts the idle timeout timer."""
        if self.idle_timeout <= 0 or self.compute_device == "cpu":
            return
        if self.timer is not None:
            self.timer.cancel()
        self.timer = threading.Timer(self.idle_timeout, self.offload)
        self.timer.daemon = True
        self.timer.start()

    def _wake_up(self):
        """Moves the model to VRAM. MUST be called from within self.lock."""
        if self.is_offloaded and self.compute_device != "cpu":
            logger.info("Waking up model to VRAM...")
            self.model.to(self.compute_device)
            self.is_offloaded = False

    def offload(self):
        """Moves the model back to RAM and clears GPU cache."""
        with self.lock:
            # Prevent race condition if unload() was called just as the timer fired
            if self.model is None:
                return

            if not self.is_offloaded and self.compute_device != "cpu":
                logger.info(f"Model idle for {self.idle_timeout}s. Offloading to RAM...")
                self.model.to("cpu")
                import torch

                if self.compute_device == "cuda":
                    torch.cuda.empty_cache()
                elif self.compute_device == "mps":
                    torch.mps.empty_cache()
                self.is_offloaded = True

    def _extract_tensor(self, features):
        import torch

        if isinstance(features, torch.Tensor):
            return features
        if hasattr(features, "pooler_output") and features.pooler_output is not None:
            return features.pooler_output
        if hasattr(features, "image_embeds") and features.image_embeds is not None:
            return features.image_embeds
        if hasattr(features, "text_embeds") and features.text_embeds is not None:
            return features.text_embeds
        if isinstance(features, tuple):
            return features[1] if len(features) > 1 else features[0]
        return features[0]

    def embed_image(self, image: Image.Image) -> np.ndarray:
        import torch

        with self.lock:
            self._wake_up()
            with torch.no_grad():
                inputs = self.processor(images=image, return_tensors="pt").to(self.compute_device)
                features = self.model.get_image_features(**inputs)
                features = self._extract_tensor(features)
                features = features / features.norm(dim=-1, keepdim=True)
                result = features.cpu().numpy().astype(self.embedding_dtype)
        self._reset_timer()
        return result

    def embed_text(self, text: str) -> np.ndarray:
        import torch

        with self.lock:
            self._wake_up()
            with torch.no_grad():
                inputs = self.processor(text=text, return_tensors="pt").to(self.compute_device)
                features = self.model.get_text_features(**inputs)
                features = self._extract_tensor(features)
                features = features / features.norm(dim=-1, keepdim=True)
                result = features.cpu().numpy().astype(self.embedding_dtype)
        self._reset_timer()
        return result

    def embed_batch(self, images: list[Image.Image], batch_size: int = 32) -> list[np.ndarray]:
        import torch

        with self.lock:
            self._wake_up()
            with torch.no_grad():
                all_embeddings = []
                for i in range(0, len(images), batch_size):
                    batch = images[i : i + batch_size]
                    inputs = self.processor(images=batch, return_tensors="pt", padding=True).to(self.compute_device)
                    features = self.model.get_image_features(**inputs)
                    features = self._extract_tensor(features)
                    features = features / features.norm(dim=-1, keepdim=True)
                    all_embeddings.extend(features.cpu().numpy().astype(self.embedding_dtype))
        self._reset_timer()
        return all_embeddings

    def unload(self):
        """
        Release model resources and free GPU memory.
        Call this before destroying the embedder or when switching models.
        """
        with self.lock:
            if self.timer is not None:
                self.timer.cancel()
                self.timer = None

            import torch

            if self.model is not None:
                del self.model
                self.model = None
            if self.processor is not None:
                del self.processor
                self.processor = None

            if self.compute_device == "cuda":
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared.")
            elif self.compute_device == "mps":
                torch.mps.empty_cache()

            logger.info("ImageEmbedder unloaded.")
