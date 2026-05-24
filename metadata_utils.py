import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ImageMetadata:
    filepath: str
    filename: str
    width: int | None = None
    height: int | None = None
    comfy_model: str | None = None        # checkpoint name, stem only (no .safetensors)
    comfy_positive_prompt: str | None = None
    comfy_sampler: str | None = None
    comfy_steps: int | None = None
    comfy_cfg: float | None = None
    comfy_scheduler: str | None = None
    has_comfy_workflow: bool = False


def get_image_metadata(filepath: str) -> ImageMetadata:
    """Return a populated ImageMetadata. Never raise — catch all exceptions internally and return partial data."""
    filename = Path(filepath).name
    metadata = ImageMetadata(filepath=filepath, filename=filename)

    try:
        # For .mp4 files: return immediately with only filename populated. Skip all PIL work.
        if filepath.lower().endswith(".mp4"):
            return metadata

        # Lazy PIL import to avoid slowing startup.
        from PIL import Image

        with Image.open(filepath) as img:
            metadata.width, metadata.height = img.size

            # ComfyUI stores the executable API graph JSON in the "prompt" key.
            raw_json = img.info.get("prompt")
            if raw_json:
                comfy_data = _parse_comfy_workflow(str(raw_json))
                if comfy_data:
                    metadata.has_comfy_workflow = True
                    metadata.comfy_model = comfy_data.get("comfy_model")
                    metadata.comfy_positive_prompt = comfy_data.get("comfy_positive_prompt")
                    metadata.comfy_sampler = comfy_data.get("comfy_sampler")
                    metadata.comfy_steps = comfy_data.get("comfy_steps")
                    metadata.comfy_cfg = comfy_data.get("comfy_cfg")
                    metadata.comfy_scheduler = comfy_data.get("comfy_scheduler")

    except Exception:
        logger.exception("Failed to extract metadata from %s", filepath)

    return metadata


def _parse_comfy_workflow(raw_json: str) -> dict:
    """Private helper. Parse the JSON string and extract ComfyUI workflow fields.

    Iterates the node graph (dict of node-id → node-object). For each node,
    dispatches on ``node["class_type"]``:

    - ``"CheckpointLoaderSimple"``: extract ``inputs["ckpt_name"]``, strip the file
      extension using ``Path(...).stem``, assign to ``comfy_model``.
    - ``"KSampler"`` or ``"KSamplerAdvanced"``: extract ``inputs["sampler_name"]``,
      ``inputs["steps"]``, ``inputs["cfg"]``, ``inputs["scheduler"]``.
    - ``"CLIPTextEncode"``: collect all instances. After iteration, if exactly one or
      two exist, treat the first encountered as the positive prompt (this is a
      best-effort heuristic). Only populate ``comfy_positive_prompt`` if the text is
      non-empty and under 1000 characters to avoid dumping massive prompt dumps into
      the UI.

    If JSON parsing fails or any key is missing, catch the exception silently and
    return whatever was successfully extracted. ``has_comfy_workflow`` is set to
    ``True`` only if at least one ComfyUI-specific node was found.
    """
    result: dict = {}
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return result

    nodes = data if isinstance(data, dict) else {}
    clip_text_encodes: list[str] = []
    found_comfy_node = False

    try:
        for node in nodes.values():
            if not isinstance(node, dict):
                continue
            class_type = node.get("class_type")
            inputs = node.get("inputs", {})

            if class_type == "CheckpointLoaderSimple":
                ckpt_name = inputs.get("ckpt_name")
                if ckpt_name and isinstance(ckpt_name, str):
                    result["comfy_model"] = Path(ckpt_name).stem
                    found_comfy_node = True

            elif class_type in ("KSampler", "KSamplerAdvanced"):
                sampler_name = inputs.get("sampler_name")
                if sampler_name and isinstance(sampler_name, str):
                    result["comfy_sampler"] = sampler_name
                    found_comfy_node = True
                steps = inputs.get("steps")
                if steps is not None and not isinstance(steps, list):
                    try:
                        result["comfy_steps"] = int(steps)
                        found_comfy_node = True
                    except (ValueError, TypeError):
                        pass
                cfg = inputs.get("cfg")
                if cfg is not None and not isinstance(cfg, list):
                    try:
                        result["comfy_cfg"] = float(cfg)
                        found_comfy_node = True
                    except (ValueError, TypeError):
                        pass
                scheduler = inputs.get("scheduler")
                if scheduler and isinstance(scheduler, str):
                    result["comfy_scheduler"] = scheduler
                    found_comfy_node = True

            elif class_type == "CLIPTextEncode":
                text = inputs.get("text")
                if text and isinstance(text, str):
                    clip_text_encodes.append(text)

        # Best-effort heuristic: if exactly one or two CLIPTextEncode nodes exist,
        # treat the first encountered as the positive prompt.
        if clip_text_encodes and len(clip_text_encodes) <= 2:
            prompt = clip_text_encodes[0]
            if prompt and len(prompt) < 1000:
                result["comfy_positive_prompt"] = prompt
                found_comfy_node = True

    except Exception:
        logger.exception("Error parsing ComfyUI workflow")

    if not found_comfy_node:
        return {}

    return result
