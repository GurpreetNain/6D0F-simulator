import sys
import os

# Ensure Python can find the src package
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from PyQt6.QtWidgets import QApplication
from src.gui.main_window import SimulationHost

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SimulationHost()
    window.show()
    sys.exit(app.exec())