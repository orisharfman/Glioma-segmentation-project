import os
import sys
import shutil
import numpy as np
from segmentation_stats import calculate_statistics, load_mask_folder_as_3d_array
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QLineEdit, QFileDialog,
    QTextEdit, QSizePolicy, QStackedWidget, QGraphicsScene,
    QGraphicsView, QGraphicsPixmapItem, QMessageBox
)
from PyQt5.QtGui import QPixmap, QFont, QMovie
from PyQt5.QtCore import Qt, QTimer

# --- Start Screen ---
class StartScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_file = None
        self.parent_window = parent

        layout = QVBoxLayout()

        # Patient info inputs
        self.patient_id_input = QLineEdit()
        self.patient_id_input.setPlaceholderText("Patient ID")
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Name")
        self.age_input = QLineEdit()
        self.age_input.setPlaceholderText("Age")
        self.gender_input = QLineEdit()
        self.gender_input.setPlaceholderText("Gender")
        self.date_input = QLineEdit()
        self.date_input.setPlaceholderText("Scan Date (YYYY-MM-DD)")

        layout.addWidget(self.patient_id_input)
        layout.addWidget(self.name_input)
        layout.addWidget(self.age_input)
        layout.addWidget(self.gender_input)
        layout.addWidget(self.date_input)

        # File chooser
        self.choose_file_button = QPushButton("Choose .nii file")
        self.file_label = QLabel("No file selected")
        self.file_label.setWordWrap(True)
        self.file_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.choose_file_button)
        layout.addWidget(self.file_label)

        self.choose_file_button.clicked.connect(self.choose_file)

        # Start segmentation button
        self.start_button = QPushButton("Start Segmentation")
        layout.addWidget(self.start_button)
        self.start_button.clicked.connect(self.start_segmentation)

        self.setLayout(layout)

    def choose_file(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Select NIfTI file", "", "NIfTI Files (*.nii *.nii.gz)"
        )
        if file_path:
            self.selected_file = file_path
            self.file_label.setText(file_path)

    def show_warning(self, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Warning")
        msg.setText(message)
        msg.exec_()

    def start_segmentation(self):
        # Check that all patient info fields are filled
        if not all([
            self.patient_id_input.text().strip(),
            self.name_input.text().strip(),
            self.age_input.text().strip(),
            self.gender_input.text().strip(),
            self.date_input.text().strip(),
        ]):
            self.show_warning("Please fill in all patient information fields before starting.")
            return

        # Also check for .nii file selected
        if not self.selected_file:
            self.show_warning("Please select a .nii file before starting.")
            return

        temp_input_path = "temp_input"
        os.makedirs(temp_input_path, exist_ok=True)

        # Clear temp_input directory
        for f in os.listdir(temp_input_path):
            try:
                os.remove(os.path.join(temp_input_path, f))
            except Exception as e:
                print(f"Error deleting file: {e}")

        # Copy selected file to temp_input
        dest_path = os.path.join(temp_input_path, os.path.basename(self.selected_file))
        shutil.copyfile(self.selected_file, dest_path)

        # Save patient info in parent for later use
        self.parent_window.patient_info = {
            "Patient ID": self.patient_id_input.text(),
            "Name": self.name_input.text(),
            "Age": self.age_input.text(),
            "Gender": self.gender_input.text(),
            "Scan Date": self.date_input.text(),
        }
        print("Patient Info:", self.parent_window.patient_info)
        print(f"File copied to: {dest_path}")

        # Switch to loading screen
        self.parent_window.stack.setCurrentIndex(1)

        # Simulate segmentation delay then switch to result screen
        QTimer.singleShot(3000, self.parent_window.finish_segmentation)



# --- Loading Screen ---
class LoadingScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        self.label = QLabel("Processing, please wait...")
        self.label.setAlignment(Qt.AlignCenter)

        # Spinner gif
        self.spinner_label = QLabel()
        self.spinner_label.setAlignment(Qt.AlignCenter)
        self.movie = QMovie("assets/spinner.gif")  # Adjust path to your spinner gif here
        self.spinner_label.setMovie(self.movie)
        self.movie.start()

        layout.addWidget(self.label)
        layout.addWidget(self.spinner_label)
        self.setLayout(layout)

# --- Result Screen ---
class ResultScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.images_folder = "temp_results"
        self.image_paths = []
        self.current_index = 0

        self.init_ui()

    def init_ui(self):
        # Image display area
        self.scene = QGraphicsScene()
        self.view_area = QGraphicsView(self.scene)
        self.view_area.setFixedSize(600, 500)
        self.view_area.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # Navigation buttons
        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.image_counter = QLabel("")
        self.prev_button.clicked.connect(self.show_prev_image)
        self.next_button.clicked.connect(self.show_next_image)

        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.image_counter)
        nav_layout.addWidget(self.next_button)

        image_layout = QVBoxLayout()
        image_layout.addWidget(self.view_area)
        image_layout.addLayout(nav_layout)

        # Patient info and statistics text areas
        self.patient_label = QLabel("<b>Patient Information</b>")
        self.patient_label.setFont(QFont("", 14))
        self.patient_text = QTextEdit()
        self.patient_text.setReadOnly(True)
        self.patient_text.setFixedWidth(300)

        self.stats_label = QLabel("<b>Statistics</b>")
        self.stats_label.setFont(QFont("", 14))
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setFixedWidth(300)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.patient_label)
        right_layout.addWidget(self.patient_text)
        right_layout.addWidget(self.stats_label)
        right_layout.addWidget(self.stats_text)

        # Bottom buttons
        self.back_button = QPushButton("Back")
        self.save_button = QPushButton("Save Segmentation")
        self.export_button = QPushButton("Export Stats as PDF")

        # Make buttons bigger
        for btn in [self.back_button, self.save_button, self.export_button]:
            btn.setFixedHeight(40)  # Increase height
            btn.setStyleSheet("font-size: 20px;")  # Optional: increase font size

        self.back_button.clicked.connect(self.back_to_start)
        self.save_button.clicked.connect(self.save_segmentation_placeholder)
        self.export_button.clicked.connect(self.export_stats_placeholder)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.back_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.export_button)

        # Main layout horizontal
        main_layout = QHBoxLayout()
        main_layout.addLayout(image_layout)
        main_layout.addLayout(right_layout)

        # Overall vertical layout
        overall_layout = QVBoxLayout()
        overall_layout.addLayout(main_layout)
        overall_layout.addLayout(button_layout)

        self.setLayout(overall_layout)

    def load_images(self):
        self.image_paths = sorted([
            os.path.join(self.images_folder, f)
            for f in os.listdir(self.images_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.current_index = 0
        self.update_image_display()

    def update_image_display(self):
        if not self.image_paths:
            self.scene.clear()
            self.image_counter.setText("No images found.")
            return

        self.current_index = max(0, min(self.current_index, len(self.image_paths) - 1))
        image_path = self.image_paths[self.current_index]
        pixmap = QPixmap(image_path)

        self.scene.clear()
        if pixmap.isNull():
            self.image_counter.setText(f"Failed to load image {os.path.basename(image_path)}")
            return

        pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(pixmap_item)
        self.view_area.fitInView(pixmap_item, Qt.KeepAspectRatio)

        self.image_counter.setText(f"Image {self.current_index + 1} of {len(self.image_paths)}")

    def show_next_image(self):
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.update_image_display()

    def show_prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_image_display()

    def set_patient_info(self, info):
        text = "\n".join(f"{k}: {v}" for k, v in info.items())
        self.patient_text.setText(text)

    def set_statistics(self, stats):
        text = "\n".join(f"{k}: {v}" for k, v in stats.items())
        self.stats_text.setText(text)

    # Placeholder button functions
    def back_to_start(self):
        self.parent_window.stack.setCurrentIndex(0)

    def save_segmentation_placeholder(self):
        print("Save segmentation pressed (placeholder)")

    def export_stats_placeholder(self):
        print("Export stats as PDF pressed (placeholder)")

# --- Main Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tumor Segmentation GUI")
        self.resize(1000, 700)

        self.patient_info = {}
        self.statistics = {
            "Total Tumor Volume (cm³)": "23.5",
            "Max Diameter (cm)": "3.7",
            "Sphericity": "0.78",
            "Slices with Tumor": "39",
        }

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.start_screen = StartScreen(self)
        self.loading_screen = LoadingScreen(self)
        self.result_screen = ResultScreen(self)

        self.stack.addWidget(self.start_screen)   # index 0
        self.stack.addWidget(self.loading_screen) # index 1
        self.stack.addWidget(self.result_screen)  # index 2

        self.stack.setCurrentIndex(0)  # Show start screen first

    def finish_segmentation(self):
        print("Segmentation finished, switching to result screen...")

        # Load images into ResultScreen
        self.result_screen.load_images()

        # Pass patient info and statistics
        self.result_screen.set_patient_info(self.patient_info)
        self.result_screen.set_statistics(self.statistics)

        self.stack.setCurrentIndex(2)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
