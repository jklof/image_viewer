# AI Image Search

The project is an "AI Image Explorer," a desktop application built with PySide6. Its core purpose is to use a CLIP model to generate embeddings for local images, store them in an SQLite database, and allow a user to perform similarity searches via text or a query image, and visualize the results.

### Core Features
*   **Text & Image Search:** Find images using text descriptions or by providing an example image.
*   **Interactive Visualization:** Explore your entire image library in a 2D map where similar images are clustered together.

---

## Getting Started

**1. Install Dependencies**

pip

```bash
pip install -r requirements.txt
```

or conda

```bash
conda env create -f environment.yml
conda activate image-viewer-dev
```

**2. Configure Image Directories**
Create a `config.yml` file in the project root and add the paths to your image folders:
```yaml
# config.yml
# --- Model Configuration ---
# The Hugging Face model ID for the CLIP model used for generating embeddings.
# WARNING: Changing this requires a full database re-sync, as embeddings
# generated with different models are incompatible.
# Recommended Models (Sorted by Increasing Size/Quality):
# 
# openai/clip-vit-base-patch32: 
#   Size: 512 dim | VRAM: ~0.5GB | Speed: Fastest | Quality: Good Baseline
# 
# laion/CLIP-ViT-B-32-laion2B-s34B-b79K: 
#   Size: 512 dim | VRAM: ~0.6GB | Speed: Very Fast | Quality: Good Baseline (Better data)
# 
# openai/clip-vit-large-patch14: 
#   Size: 768 dim | VRAM: ~1.3GB | Speed: Fast | Quality: Excellent (Commonly used default)
# 
# laion/CLIP-ViT-L-14-laion2B-s32B-b82K: 
#   Size: 768 dim | VRAM: ~1.5GB | Speed: Moderate | Quality: Excellent (LAION-trained default)
# 
# laion/CLIP-ViT-H-14-laion2B-s32B-b79K:
#   Size: 1024 dim | VRAM: ~3.5GB | Speed: Slower | Quality: Top Tier
model_id: "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"

# Location where the database file will be written to
database_path: "image.db"

# --- Directory Configuration ---
# List the full paths to the directories you want to scan for images.
# The 'sync' command will use these directories to keep the database up-to-date.
directories:
  - "/path/to/your/photos"
  - "C:/Users/YourName/Pictures"
```

**3. Index Your Images**
Run the command-line tool to scan your files. This may take a while on the first run.
```bash
python image_cli.py sync
```

**4. Launch the App**
```bash
python main.py
```

