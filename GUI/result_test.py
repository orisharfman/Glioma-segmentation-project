import os
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGraphicsScene, QGraphicsView,
    QGraphicsPixmapItem, QTextEdit, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt


class ResultScreen(QWidget):
    def __init__(self, patient_info, statistics, images_folder='temp_results'):
        super().__init__()
        self.setWindowTitle("Segmentation Results")
        self.resize(1000, 700)

        self.patient_info = patient_info
        self.statistics = statistics
        self.images_folder = images_folder

        # Load image paths (only PNG/JPG)
        self.image_paths = sorted([
            os.path.join(images_folder, f)
            for f in os.listdir(images_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.current_index = 0

        self.init_ui()

    def init_ui(self):
        # Graphics view for image display
        self.scene = QGraphicsScene()
        self.view_area = QGraphicsView(self.scene)
        self.view_area.setFixedSize(600, 500)

        # Image navigation controls
        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.image_counter = QLabel("")
        self.image_counter.setAlignment(Qt.AlignCenter)

        self.prev_button.clicked.connect(self.show_prev_image)
        self.next_button.clicked.connect(self.show_next_image)

        nav_layout = QHBoxLayout()
        nav_layout.addStretch()
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.image_counter)
        nav_layout.addWidget(self.next_button)
        nav_layout.addStretch()

        image_layout = QVBoxLayout()
        image_layout.addWidget(self.view_area)
        image_layout.addLayout(nav_layout)

        # Info and stats display
        font = QFont()
        font.setPointSize(12)

        self.patient_text = QTextEdit()
        self.patient_text.setReadOnly(True)
        self.patient_text.setFont(font)
        self.patient_text.setMinimumWidth(300)
        self.patient_text.setText(self.format_patient_info())

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setFont(font)
        self.stats_text.setMinimumWidth(300)
        self.stats_text.setText(self.format_statistics())

        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("<b>Patient Information</b>"))
        right_layout.addWidget(self.patient_text)
        right_layout.addWidget(QLabel("<b>Statistics</b>"))
        right_layout.addWidget(self.stats_text)
        right_layout.addStretch()

        # Horizontal layout: image on left, info on right
        content_layout = QHBoxLayout()
        content_layout.addLayout(image_layout, 2)
        content_layout.addLayout(right_layout, 1)

        # Bottom action buttons
        self.back_button = QPushButton("Back")
        self.save_button = QPushButton("Save Segmentation")
        self.export_button = QPushButton("Export Stats as PDF")

        self.back_button.clicked.connect(self.on_back_clicked)
        self.save_button.clicked.connect(self.on_save_clicked)
        self.export_button.clicked.connect(self.on_export_clicked)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.back_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.export_button)
        button_layout.addStretch()

        # Final layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(content_layout)
        main_layout.addSpacing(10)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)
        self.update_image_display()

    def format_patient_info(self):
        return "\n".join(f"{k}: {v}" for k, v in self.patient_info.items())

    def format_statistics(self):
        return "\n".join(f"{k}: {v}" for k, v in self.statistics.items())

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
            self.image_counter.setText(f"Failed to load image: {os.path.basename(image_path)}")
            return

        item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(item)
        self.view_area.fitInView(item, Qt.KeepAspectRatio)
        self.image_counter.setText(f"Image {self.current_index + 1} of {len(self.image_paths)}")

    def show_next_image(self):
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.update_image_display()

    def show_prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_image_display()

    def on_back_clicked(self):
        print("Back clicked (placeholder)")

    def on_save_clicked(self):
        print("Save Segmentation clicked (placeholder)")

    def on_export_clicked(self):
        print("Export Stats as PDF clicked (placeholder)")


if __name__ == "__main__":
    patient_info = {
        "Patient ID": "123456789",
        "Name": "John Doe",
        "Age": "45",
        "Gender": "Male",
        "Scan Date": "2025-05-24"
    }
    statistics = {
        "Total Tumor Volume (cm³)": "23.5",
        "Max Diameter (cm)": "3.7",
        "Sphericity": "0.78",
        "Slices with Tumor": "39",
    }

    app = QApplication(sys.argv)
    window = ResultScreen(patient_info, statistics)
    window.show()
    sys.exit(app.exec_())
