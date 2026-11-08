"""
Medical DICOM Workstation & 3D Volume Reconstruction
Application Entry Point.
"""
import sys
import os

# Ensure current project directory is in python search path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon, QPalette, QColor
from PySide6.QtCore import Qt

from ui.main_window import MainWindow


def main():
    # High-DPI display scaling
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

    app = QApplication(sys.argv)
    app.setApplicationName("Antigravity Medical DICOM Workstation")
    app.setOrganizationName("Antigravity Medical Imaging")

    # Set dark Fusion theme
    app.setStyle("Fusion")
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(20, 20, 22))
    dark_palette.setColor(QPalette.WindowText, QColor(224, 224, 224))
    dark_palette.setColor(QPalette.Base, QColor(15, 15, 18))
    dark_palette.setColor(QPalette.AlternateBase, QColor(25, 25, 28))
    dark_palette.setColor(QPalette.ToolTipBase, QColor(40, 40, 48))
    dark_palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.Text, QColor(224, 224, 224))
    dark_palette.setColor(QPalette.Button, QColor(32, 32, 36))
    dark_palette.setColor(QPalette.ButtonText, QColor(224, 224, 224))
    dark_palette.setColor(QPalette.BrightText, QColor(0, 210, 255))
    dark_palette.setColor(QPalette.Link, QColor(0, 150, 255))
    dark_palette.setColor(QPalette.Highlight, QColor(0, 122, 204))
    dark_palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(dark_palette)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
