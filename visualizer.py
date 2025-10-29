# visualizer.py

import http.server
import socketserver
import threading
import webbrowser
import urllib.parse
from io import BytesIO
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from PIL import Image

# We assume these are installed if the user runs the visualize command.
import umap
import hdbscan

# --- Module-level state for the server ---
VALID_THUMBNAIL_PATHS = set()

# This is the JavaScript code we will inject into the HTML file.
# It creates and manages a custom tooltip div.
CUSTOM_JS = """
<script>
    document.addEventListener('DOMContentLoaded', function() {
        var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
        
        var tooltip = document.createElement('div');
        tooltip.id = 'custom-tooltip';
        tooltip.style.position = 'absolute';
        tooltip.style.display = 'none';
        tooltip.style.border = '1px solid #ccc';
        tooltip.style.backgroundColor = 'rgba(255, 255, 255, 0.9)';
        tooltip.style.padding = '10px';
        tooltip.style.borderRadius = '5px';
        tooltip.style.pointerEvents = 'none';
        tooltip.style.zIndex = '1000';
        document.body.appendChild(tooltip);

        plotDiv.on('plotly_hover', function(data) {
            if (data.points.length > 0) {
                var point = data.points[0];
                var customData = point.customdata;
                var imageUrl = customData[0];
                var filePath = customData[1];
                var clusterInfo = customData[2];

                tooltip.innerHTML = `
                    <img src="${imageUrl}" width="150"><br>
                    <b>Path:</b> ${filePath}<br>
                    ${clusterInfo}
                `;
                
                var event = window.event || arguments[0];
                tooltip.style.left = (event.pageX + 15) + 'px';
                tooltip.style.top = (event.pageY + 15) + 'px';
                tooltip.style.display = 'block';
            }
        });

        plotDiv.on('plotly_unhover', function(data) {
            tooltip.style.display = 'none';
        });
    });
</script>
"""


class ThumbnailRequestHandler(http.server.SimpleHTTPRequestHandler):
    """A custom request handler that serves the main HTML and dynamic thumbnails."""

    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        if parsed_path.path == "/thumbnail":
            self.serve_thumbnail(parsed_path)
        else:
            super().do_GET()

    def serve_thumbnail(self, parsed_path):
        params = urllib.parse.parse_qs(parsed_path.query)
        filepath_str = params.get("path", [None])[0]
        if not filepath_str:
            self.send_error(400, "Missing 'path' parameter")
            return

        filepath = Path(urllib.parse.unquote(filepath_str))
        if str(filepath.resolve()) not in VALID_THUMBNAIL_PATHS:
            self.send_error(404, "File not found or not authorized")
            return

        try:
            with Image.open(filepath) as img:
                img.thumbnail((150, 150))
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                self.send_response(200)
                self.send_header("Content-type", "image/jpeg")
                self.end_headers()
                self.wfile.write(buffer.getvalue())
        except Exception as e:
            self.send_error(500, f"Error generating thumbnail: {e}")


def _create_visualization_html(embedding_data: list, output_path: str, host: str, port: int):
    """Internal function to generate the plot HTML file."""
    print(f"Processing {len(embedding_data)} unique images for visualization.")
    all_embeddings = np.array([item["embedding"] for item in embedding_data])

    print("Performing dimensionality reduction with UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    coords_2d = reducer.fit_transform(all_embeddings)

    print("Clustering embeddings with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=None)
    cluster_labels = clusterer.fit_predict(coords_2d)

    print("Generating interactive visualization HTML...")

    thumbnail_urls, filepath_texts, cluster_texts = [], [], []
    for i, item in enumerate(embedding_data):
        filepath = item["filepath"]
        encoded_path = urllib.parse.quote(filepath)
        full_url = f"http://{host}:{port}/thumbnail?path={encoded_path}"
        thumbnail_urls.append(full_url)
        filepath_texts.append(filepath)
        cluster_texts.append(f"Cluster: {cluster_labels[i]}")

    custom_data = np.stack((thumbnail_urls, filepath_texts, cluster_texts), axis=-1)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=coords_2d[:, 0],
            y=coords_2d[:, 1],
            mode="markers",
            marker=dict(
                color=cluster_labels,
                colorscale="Rainbow",
                showscale=True,
                size=8,
                opacity=0.8,
                colorbar=dict(title="Cluster ID"),
            ),
            customdata=custom_data,
            hoverinfo="none",
        )
    )
    fig.update_layout(
        title="Interactive 2D Visualization of Image Embeddings",
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        hovermode="closest",
        template="plotly_dark",
    )

    html_content = fig.to_html(full_html=True, include_plotlyjs="cdn")
    final_html = html_content + CUSTOM_JS
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_html)
    print("HTML file created successfully.")


def _run_server(html_file_path: str, host: str, port: int):
    """Internal function to start the web server."""
    directory = Path(html_file_path).parent

    def handler_factory(*args, **kwargs):
        return ThumbnailRequestHandler(*args, directory=str(directory), **kwargs)

    with socketserver.TCPServer((host, port), handler_factory) as httpd:
        url = f"http://{host}:{port}/{Path(html_file_path).name}"
        print(f"\n--- Visualization Server Running ---")
        print(f"Opening visualization at: {url}")
        print("Press Ctrl+C to stop the server.")
        threading.Timer(1, lambda: webbrowser.open_new_tab(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down the server.")
            httpd.shutdown()


def start_visualization_server(embedding_data: list, output_html: str, host: str, port: int):
    """The main entry point for the visualization module."""
    if not embedding_data or len(embedding_data) < 3:
        print("Error: Need at least 3 images to generate a visualization.")
        return

    # Populate the global set of valid paths for the security check
    global VALID_THUMBNAIL_PATHS
    VALID_THUMBNAIL_PATHS = {str(Path(item["filepath"]).resolve()) for item in embedding_data}

    # 1. Generate the HTML file
    _create_visualization_html(embedding_data, output_html, host, port)

    # 2. Start the server to view it
    _run_server(output_html, host, port)
