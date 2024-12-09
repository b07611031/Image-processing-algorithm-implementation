import sys
import numpy as np
import cv2 as cv
# from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import Ui_hw1_gui as ui
import img_processing as ip


class MainWindow(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.img1.clicked.connect(self.img1_open)
        self.img2.clicked.connect(self.img2_open)
        self.output.clicked.connect(self.save_output)
        self.sign_equal.clicked.connect(self.operator)
        self.aug_4.clicked.connect(self.formula)
        self.aug_mixup.clicked.connect(self.mixup)
        self.actionOpen_Image_1.triggered.connect(self.img1_open)
        self.actionOpen_Image_2.triggered.connect(self.img2_open)
        self.actionSave_Output.triggered.connect(self.save_output)
        self.actionClose.triggered.connect(lambda: self.close())

    # Open Image1
    def img1_open(self):
        file_path = QFileDialog.getOpenFileName(self, "Folder")[0]
        self.img1_path = file_path
        # print(file_path)
        if not file_path.endswith('.64'):
            return
        self.img1_arr = ip.read_b64(file_path)
        self.intput_arr = self.img1_arr
        self.img1_show.setPixmap(self.show_image(self.img1_arr))
        self.img1_show.setScaledContents(True)
        self.img1.adjustSize()
        self.img1_hist_path = ip.plot_hist(self.img1_arr, file_path)
        self.img1_hist.setPixmap(QPixmap(self.img1_hist_path))
        self.img1_hist.setScaledContents(True)

    # Open Image2
    def img2_open(self):
        file_path = QFileDialog.getOpenFileName(self, "Folder")[0]
        if not file_path.endswith('.64'):
            return
        self.img2_arr = ip.read_b64(file_path)
        self.intput_arr = self.img2_arr
        self.img2_show.setPixmap(self.show_image(self.img2_arr))
        self.img2_show.setScaledContents(True)
        self.img2.adjustSize()
        self.img2_hist_path = ip.plot_hist(self.img2_arr, file_path)
        self.img2_hist.setPixmap(QPixmap(self.img2_hist_path))
        self.img2_hist.setScaledContents(True)

    def operator(self):
        multi = self.input_multiply.value()*255/31
        divide = self.input_divide.value()*255/31
        add = self.input_add.value()*255/31
        subtract = self.input_subtract.value()*255/31
        # If have two image.
        try:
            switch = {0: self.img1_arr.copy(), 1: self.img2_arr.copy()}
            self.output_arr = switch[self.comboBox.currentIndex(
            )] * multi / divide + add - subtract
        except:
            # If have only one image, it proceed the one it have.
            try:
                self.output_arr = self.intput_arr.copy() * multi / divide + add - subtract
            # Don't have any img, nothing happen.
            except:
                return
        self.show_ouput()
        
    # image subtract the image of shift to left one pixel.
    def formula(self):
        try:
            switch = {0: self.img1_arr.copy(), 1: self.img2_arr.copy()}
            self.output_arr = switch[self.comboBox.currentIndex(
            )]
        except:
            try:
                self.output_arr = self.intput_arr.copy()
            except:
                return
        left_image = np.zeros(self.output_arr.shape)
        for i in range(self.output_arr.shape[0]):
            for j in range(self.output_arr.shape[1]):
                left_image[i, j] = self.output_arr[i, j-1]
        self.output_arr -= left_image
        self.show_ouput()

    def mixup(self):
        try:
            self.output_arr = (self.img1_arr+self.img2_arr)/2
        except:
            try:
                self.output_arr = self.intput_arr.copy()
            except:
                return
        self.show_ouput()

    def show_image(self, array):
        Qimage = QImage(np.uint8(
            array), array.shape[1], array.shape[0], array.shape[0], QImage.Format_Grayscale8)
        Qimage = QPixmap(Qimage)
        return Qimage
    
    def save_output(self):
        try:
            cv.imwrite('./Output.png', self.output_arr)
            print('Output saved successfully.')
        except:
            pass
    
    def show_ouput(self):
        self.output_arr[self.output_arr > 255] = 255
        self.output_arr[self.output_arr < 0] = 0
        self.output_show.setPixmap(self.show_image(self.output_arr))
        self.output_show.setScaledContents(True)
        self.output.adjustSize()
        self.output_hist_path = ip.plot_hist(self.output_arr)
        self.output_hist.setPixmap(QPixmap(self.output_hist_path))
        self.output_hist.setScaledContents(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
