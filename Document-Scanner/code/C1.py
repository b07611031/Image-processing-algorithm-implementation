import sys
import numpy as np
import cv2 as cv
import time
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import Ui_C1_gui as ui
import img_processing as ip


class MainWindow(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # self.img_open()
        self.open_button.clicked.connect(self.img_open)
        self.actionQuit.triggered.connect(lambda: self.close())
        self.actionOpen_file.triggered.connect(self.img_open)

        self.bright_slider.valueChanged.connect(self.bright_slider_display)
        self.contrast_slider.valueChanged.connect(self.contrast_slider_display)
        self.bright_checkbox.stateChanged.connect(self.img_aug)
        self.contrast_checkbox.stateChanged.connect(self.img_aug)

        self.rotate_button.clicked.connect(self.rotate)
        self.find_button.clicked.connect(self.find)
        self.crop_button.clicked.connect(self.crop)
        self.crop_button_2.clicked.connect(self.crop2)

    # Open Image
    def img_open(self):
        # self.img_path = './img/doc(1).jpg'
        self.img_path = QFileDialog.getOpenFileName(
            self, "Open image file", '', '*.bmp;*.jpg;*.JPG;*.png')[0]
        if self.img_path == '':
            return
        self.img = cv.imdecode(
            np.fromfile(self.img_path, dtype=np.uint8), 1)
        h, w = self.img.shape[0], self.img.shape[1]
        if h > 600 or w > 600:
            self.img = cv.resize(self.img, (int(w*594/h), 594), interpolation=cv.INTER_AREA)
        self.open_button.adjustSize()
        self.path_label.setText(self.img_path.split('/')[-1])
        self.size_label.setText(
            f'({self.img.shape[0]}, {self.img.shape[1]})')
        self.show_input(self.img_path)
        self.img_aug()

    def contrast_slider_display(self):
        _translate = QtCore.QCoreApplication.translate
        self.contrast_label.setText(_translate(
            "MainWindow", f"Contrast: {self.contrast_slider.value()}"))
        if self.contrast_checkbox.isChecked():
            self.img_aug()

    def bright_slider_display(self):
        _translate = QtCore.QCoreApplication.translate
        self.bright_label.setText(_translate(
            "MainWindow", f"Brightness: {self.bright_slider.value()}"))
        if self.bright_checkbox.isChecked():
            self.img_aug()

    def img_aug(self):
        brightness, c = 0, 0
        if self.bright_checkbox.isChecked():
            brightness = self.bright_slider.value()
        if self.contrast_checkbox.isChecked():
            c = self.contrast_slider.value()
        self.output_path = ip.aug(self.img, self.img_path, brightness, c)
        self.show_output(self.output_path)
        if self.find_button.isChecked():
            self.find()

    def rotate(self):
        self.img, output_path = ip.rotate(self.img, self.img_path)
        self.show_input(output_path)
        self.img_aug()

    def find(self):
        output_path = ip.scan(self.output_path, func='find')
        self.show_output(output_path)

    def crop(self):
        output_path = ip.scan(self.output_path)
        self.show_output(output_path)

    def crop2(self):
        output_path = ip.scan(self.output_path, func='binarize')
        self.show_output(output_path)

    def qmap_img2(self, img_path):
        qmap = QPixmap(img_path)
        return qmap.scaled(self.img2_show.width(), self.img2_show.height(), QtCore.Qt.KeepAspectRatio)

    def show_input(self, img_path):
        self.img2_show.setPixmap(QPixmap(img_path).scaled(
            self.img2_show.width(), self.img2_show.height(), QtCore.Qt.KeepAspectRatio))

    def show_output(self, img_path):
        self.img_show.setPixmap(QPixmap(img_path).scaled(
            self.img_show.width(), self.img_show.height(), QtCore.Qt.KeepAspectRatio))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
