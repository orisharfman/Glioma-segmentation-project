from PyQt5.QtWidgets import QDialog, QLabel, QVBoxLayout
from PyQt5.QtGui import QMovie

class LoadingWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Running Segmentation")
        self.setModal(True)
        self.setFixedSize(300, 300)

        layout = QVBoxLayout()
        self.spinner_label = QLabel(self)
        self.spinner_label.setAlignment(Qt.AlignCenter)

        self.spinner = QMovie("assets/spinner.gif")
        self.spinner_label.setMovie(self.spinner)
        self.spinner.start()

        layout.addWidget(self.spinner_label)
        self.setLayout(layout)
