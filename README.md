# Image Search

A Qt App to search your local image files using natural language or by image similarity.

### Core Features
*   **Text & Image Search:** Find images using text descriptions or by providing an example image.
*   **Interactive Visualization:** Explore your entire image library in a 2D map where similar images are clustered together.

---

## Getting Started

**1. Install Dependencies**
pip install -r requirements.txt

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