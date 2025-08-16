import sys
import os
import shutil
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QLineEdit, QFileDialog, QDialog
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QMovie



class LoadingScreen(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        self.spinner_label = QLabel(self)
        self.spinner_label.setAlignment(Qt.AlignCenter)

        # Correct and safe path to the GIF
        spinner_path = os.path.join(os.path.dirname(__file__), "assets", "spinner.gif")
        self.movie = QMovie(spinner_path)  # Store QMovie in self to prevent GC

        if not self.movie.isValid():
            print("❌ Failed to load spinner.gif. Check the path or file format.")

        self.spinner_label.setMovie(self.movie)
        self.movie.start()

        layout.addWidget(self.spinner_label)
        self.setLayout(layout)

class StartScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Start Segmentation")
        self.resize(500, 400)

        self.selected_file = ""

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Title
        title = QLabel("Fill Patient Info")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)

        # Patient info fields
        self.patient_id_input = self._add_labeled_input(layout, "Patient ID:")
        self.name_input = self._add_labeled_input(layout, "Name:")
        self.age_input = self._add_labeled_input(layout, "Age:")
        self.gender_input = self._add_labeled_input(layout, "Gender:")
        self.date_input = self._add_labeled_input(layout, "Scan Date:")

        # File chooser
        choose_file_btn = QPushButton("Choose File")
        choose_file_btn.clicked.connect(self.choose_file)
        layout.addWidget(choose_file_btn)

        self.file_label = QLabel("No file selected")
        self.file_label.setWordWrap(True)
        layout.addWidget(self.file_label)

        # Start segmentation
        start_btn = QPushButton("Start Segmentation")
        start_btn.clicked.connect(self.start_segmentation)
        layout.addWidget(start_btn)

        layout.addStretch()
        self.setLayout(layout)

    def _add_labeled_input(self, parent_layout, label_text):
        layout = QHBoxLayout()
        label = QLabel(label_text)
        line_edit = QLineEdit()
        layout.addWidget(label)
        layout.addWidget(line_edit)
        parent_layout.addLayout(layout)
        return line_edit

    def choose_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select .nii file", "", "NIfTI Files (*.nii *.nii.gz)")
        if file_path:
            self.selected_file = file_path
            self.file_label.setText(f"Selected: {file_path}")
        else:
            self.file_label.setText("No file selected")

    def start_segmentation(self):
        # Save .nii file to temp_input
        if not self.selected_file:
            self.file_label.setText("Please select a .nii file before starting.")
            return

        temp_input_path = "temp_input"
        os.makedirs(temp_input_path, exist_ok=True)

        # Clear the temp_input directory
        for f in os.listdir(temp_input_path):
            file_path = os.path.join(temp_input_path, f)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

        # Copy the new file
        dest_path = os.path.join(temp_input_path, os.path.basename(self.selected_file))
        shutil.copyfile(self.selected_file, dest_path)

        patient_info = {
            "Patient ID": self.patient_id_input.text(),
            "Name": self.name_input.text(),
            "Age": self.age_input.text(),
            "Gender": self.gender_input.text(),
            "Scan Date": self.date_input.text(),
        }

        print("Patient Info:", patient_info)
        print(f"File copied to: {dest_path}")
        # You can continue to segmentation or show loading/result screen here
        # Replace StartScreen with LoadingScreen
        self.loading_screen = LoadingScreen()
        main_window = self.window()  # Safely get top-level window
        print("Switching to loading screen...")
        main_window.setCentralWidget(self.loading_screen)
        print("Done switching. Spinner should start.")
        QTimer.singleShot(3000, self.finish_segmentation)  # simulate delay

    def finish_segmentation(self):
        self.loading_screen.done(0)  # close loading
        # Open result screen or whatever comes next




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StartScreen()
    window.show()
    sys.exit(app.exec_())
