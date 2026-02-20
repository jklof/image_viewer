import logging
import sys
import argparse
import multiprocessing

from PySide6.QtWidgets import QApplication

from app_controller import AppController
from main_window import MainWindow
from loader_manager import loader_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="AI Image Explorer")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU usage.")
    args = parser.parse_args()

    # Note: multiprocessing start method is set inside __main__ block below
    # to avoid RuntimeError when this module is imported by other scripts.

    app = QApplication(sys.argv)
    try:
        import qdarktheme

        app.setStyleSheet(qdarktheme.load_stylesheet())
    except ImportError:
        logger.info("For a dark theme, install with: pip install pyqtdarktheme")

    window = MainWindow()

    # --- INITIALIZE LOADER ---
    # Must be done after QApplication is created so QTimer can start.
    loader_manager.initialize()

    controller = AppController(main_window=window, use_cpu_only=args.cpu_only)
    controller.initialize_app()
    window.show()
    exit_code = app.exec()
    loader_manager.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    # --- CRITICAL: Set the start method to 'spawn' for CUDA safety ---
    # This must be inside __main__ to avoid RuntimeError when imported by other scripts.
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)

    main()
