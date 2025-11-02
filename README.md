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

