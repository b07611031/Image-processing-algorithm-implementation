import sys
import numpy as np
import cv2
import time
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import Ui_hw6_gui as ui
import img_processing as ip


class MainWindow(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setButtonGroup()
        # self.img1_open()
        self.img1_button.clicked.connect(self.img1_open)
        self.fimg1_button.clicked.connect(self.fimg1_open)
        self.fimg2_button.clicked.connect(self.fimg2_open)
        self.fimg3_button.clicked.connect(self.fimg3_open)
        self.fusion_button.clicked.connect(self.fus)
        self.hough_button.clicked.connect(self.hou)

        self.trap_button.clicked.connect(self.trap)
        self.wavy_button.clicked.connect(self.wav)
        self.circ_button.clicked.connect(self.circ)

        self.actionClose.triggered.connect(lambda: self.close())
        self.actionOpen_img.triggered.connect(self.img1_open)

    def setButtonGroup(self):
        self.buttongroup = QButtonGroup(self)
        self.buttongroup.addButton(self.trap_button)
        self.buttongroup.addButton(self.wavy_button)
        self.buttongroup.addButton(self.circ_button)
        self.fimg3 = None

    # Open Image1
    def img1_open(self):
        # self.img_path = './Part 1 Image/IP_dog.bmp'
        self.img_path = QFileDialog.getOpenFileName(
            self, "Open image file", '', '*.bmp;*.jpg;*.JPG;*.png')[0]
        if self.img_path == '':
            return
        img = cv2.imdecode(
            np.fromfile(self.img_path, dtype=np.uint8), 1)
        self.img1_button.adjustSize()
        self.img = ip.gray_luma(img, self.img_path)
        self.img1_show.setPixmap(self.qmap_img(self.img_path))
        self.label_size.setText(
            f'({self.img.shape[0]}, {self.img.shape[1]})')

    def trap(self):
        self.show_output(ip.trapezoidal(self.img, self.img_path))

    def wav(self):
        self.show_output(ip.wavy(self.img, self.img_path))

    def circ(self):
        self.show_output(ip.circular(self.img, self.img_path))

    def fimg1_open(self):
        # self.fimg1_path = './Part 2 Images/Image Set 1/clock1.JPG'
        self.fimg1_path = QFileDialog.getOpenFileName(
            self, "Open image file", '', '*.bmp;*.jpg;*.JPG;*.png')[0]
        if self.fimg1_path == '':
            return
        img = cv2.imdecode(
            np.fromfile(self.fimg1_path, dtype=np.uint8), 1)
        self.fimg1 = ip.gray_luma(img, self.fimg1_path)
        self.fimg1_show.setPixmap(self.qmap_img(self.fimg1_path))

    def fimg2_open(self):
        # self.fimg2_path = './Part 2 Images/Image Set 1/clock2.JPG'
        self.fimg2_path = QFileDialog.getOpenFileName(
            self, "Open image file", '', '*.bmp;*.jpg;*.JPG;*.png')[0]
        if self.fimg2_path == '':
            return
        img = cv2.imdecode(
            np.fromfile(self.fimg2_path, dtype=np.uint8), 1)
        self.fimg2 = ip.gray_luma(img, self.fimg2_path)
        self.fimg2_show.setPixmap(self.qmap_img(self.fimg2_path))

    def fimg3_open(self):
        # self.fimg3_path = './Part 2 Images/Image Set 1/clock1.JPG'
        self.fimg3_path = QFileDialog.getOpenFileName(
            self, "Open image file", '', '*.bmp;*.jpg;*.JPG;*.png')[0]
        if self.fimg3_path == '':
            return
        img = cv2.imdecode(
            np.fromfile(self.fimg3_path, dtype=np.uint8), 1)
        self.fimg3 = ip.gray_luma(img, self.fimg3_path)
        self.fimg3_show.setPixmap(self.qmap_img(self.fimg3_path))

    def fus(self):
        output_path = ip.fusion(self.fimg1, self.fimg2,
                                self.fimg3, self.fimg1_path)
        self.show_output(output_path)

    def hou(self):
        output_path = ip.hough(self.img, self.img_path)
        self.show_output(output_path)

    def qmap_img(self, img_path):
        qmap = QPixmap(img_path)
        return qmap.scaled(self.img1_show.width(), self.img1_show.height(), QtCore.Qt.KeepAspectRatio)

    def show_output(self, img_path):
        self.img2_show.setPixmap(QPixmap(img_path).scaled(
            self.img2_show.width(), self.img2_show.height(), QtCore.Qt.KeepAspectRatio))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
