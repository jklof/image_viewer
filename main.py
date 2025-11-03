import logging
import sys
import argparse

from PySide6.QtWidgets import QApplication

from app_controller import AppController
from main_window import MainWindow
from loader_manager import loader_manager

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """The main entry point for the application."""
    parser = argparse.ArgumentParser(description="AI Image Explorer")
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force the application to use the CPU for all ML computations, ignoring the GPU.",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)

    # Apply a dark theme if available
    try:
        import qdarktheme

        app.setStyleSheet(qdarktheme.load_stylesheet())
    except ImportError:
        logger.info("For a dark theme, install with: pip install pyqtdarktheme")

    # --- Composition Root ---
    # 1. Create the view (MainWindow)
    window = MainWindow()

    # 2. Create the controller and inject the view dependency and cpu-only setting
    controller = AppController(main_window=window, use_cpu_only=args.cpu_only)

    # 3. Initialize the application logic
    controller.initialize_app()

    # 4. Show the main window and start the event loop
    window.show()
    exit_code = app.exec()

    # 5. Cleanly shut down background services after the app loop exits
    logger.info("Application event loop finished. Shutting down loader manager.")
    loader_manager.shutdown()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
