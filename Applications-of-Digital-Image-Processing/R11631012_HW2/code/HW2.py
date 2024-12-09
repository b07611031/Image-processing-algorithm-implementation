import sys
import numpy as np
import cv2
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import Ui_hw2_gui as ui
import img_processing as ip


class MainWindow(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.img1_botton.clicked.connect(self.img1_open)
        self.img2_botton.clicked.connect(self.img2_open)
        self.thresh_slider.valueChanged.connect(self.thresh_slider_display)
        self.glevel_slider.valueChanged.connect(self.glevel_slider_display)
        self.resize_slider.valueChanged.connect(self.resize_slider_display)
        self.contrast_slider.valueChanged.connect(self.contrast_slider_display)
        self.bright_slider.valueChanged.connect(self.bright_slider_display)
        self.thresh_button.toggled.connect(self.thresh)
        self.glevel_button.toggled.connect(self.glevel)
        self.resize_button.toggled.connect(self.resize_img)
        self.contrast_button.toggled.connect(self.contrast)
        self.bright_button.toggled.connect(self.bright)
        self.auto_button.toggled.connect(self.equilization)
        self.actionClose.triggered.connect(lambda: self.close())

    def check(self):
        if self.thresh_button.isChecked():
            self.thresh()
        if self.glevel_button.isChecked():
            self.glevel()
        if self.resize_button.isChecked():
            self.resize_img()
        if self.contrast_button.isChecked():
            self.contrast()
        if self.bright_button.isChecked():
            self.bright()

    def thresh(self):
        threshold = self.thresh_slider.value()
        self.output_arr = self.img2_switch[self.comboBox.currentIndex()]
        self.output_arr = np.where(self.output_arr > threshold, 255, 0)
        self.show_ouput()

    def thresh_slider_display(self):
        _translate = QtCore.QCoreApplication.translate
        self.thresh_label.setText(_translate(
            "MainWindow", f"Threshold: {self.thresh_slider.value()}"))
        if self.thresh_button.isChecked():
            self.thresh()

    def glevel(self):
        gray_level = self.glevel_slider.value()
        self.output_arr = self.img2_switch[self.comboBox.currentIndex()]
        self.output_arr = np.round(
            self.output_arr/pow(2, 8-gray_level))*pow(2, 8-gray_level)
        self.show_ouput()

    def glevel_slider_display(self):
        _translate = QtCore.QCoreApplication.translate
        self.glevel_label.setText(_translate(
            "MainWindow", f"Gray level: {self.glevel_slider.value()}"))
        if self.glevel_button.isChecked():
            self.glevel()

    def resize_img(self):
        img = self.img2_switch[self.comboBox.currentIndex()]
        scale = pow(2, self.resize_slider.value()/1.2)
        row_num = round(img.shape[0]*scale)
        col_num = round(img.shape[1]*scale)
        self.output_arr = np.zeros((row_num, col_num))
        for i in range(row_num):
            for j in range(col_num):
                rf = i/scale
                cf = j/scale
                intr = int(rf)
                intc = int(cf)
                delr = rf - intr
                delc = cf - intc
                self.output_arr[i, j] = img[intr-1, intc-1]*(1-delr)*(1-delc) +\
                    img[intr, intc-1]*delr*(1-delc)+img[intr-1, intc]*(1-delr)*delc +\
                    img[intr, intc]*delr*delc
        self.show_ouput()

    def resize_slider_display(self):
        _translate = QtCore.QCoreApplication.translate
        self.resize_label.setText(_translate(
            "MainWindow", f"Resize: {self.resize_slider.value()}"))
        if self.resize_button.isChecked():
            self.resize_img()

    def contrast(self):
        c = self.contrast_slider.value()
        self.output_arr = self.img2_switch[self.comboBox.currentIndex()]
        self.output_arr = (self.output_arr-255/2) * pow(2, c/4) + 255/2
        self.show_ouput()

    def contrast_slider_display(self):
        _translate = QtCore.QCoreApplication.translate
        self.contrast_label.setText(_translate(
            "MainWindow", f"Contrast: {self.contrast_slider.value()}"))
        if self.contrast_button.isChecked():
            self.contrast()

    def bright(self):
        brightness = self.bright_slider.value()
        self.output_arr = self.img2_switch[self.comboBox.currentIndex()]
        self.output_arr = self.output_arr + brightness
        self.show_ouput()

    def bright_slider_display(self):
        _translate = QtCore.QCoreApplication.translate
        self.bright_label.setText(_translate(
            "MainWindow", f"Brightness: {self.bright_slider.value()}"))
        if self.bright_button.isChecked():
            self.bright()

    # 7
    def equilization(self):
        self.output_arr = ip.imequalize(
            self.img2_switch[self.comboBox.currentIndex()])
        self.show_ouput()

    # Open Image1
    def img1_open(self):
        file_path = QFileDialog.getOpenFileName(self, "Folder")[0]
        self.img1_path = file_path
        if not file_path.endswith('.jpg') or file_path.endswith('.png') or file_path.endswith('.BMP'):
            return
        img1 = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), 1)
        self.img1_show.setPixmap(QPixmap(file_path))
        self.img1_show.setScaledContents(True)
        self.img1_botton.adjustSize()

        imgA = self.gray_A(img1)
        self.imgA_show.setPixmap(self.show_image(imgA))
        self.imgA_show.setScaledContents(True)
        imgA_hist_path = ip.plot_hist(imgA, file_path, 'A')
        self.imgA_hist.setPixmap(QPixmap(imgA_hist_path))
        self.imgA_hist.setScaledContents(True)

        imgB = self.gray_B(img1)
        self.imgB_show.setPixmap(self.show_image(imgB))
        self.imgB_show.setScaledContents(True)
        imgB_hist_path = ip.plot_hist(imgB, file_path, 'B')
        self.imgB_hist.setPixmap(QPixmap(imgB_hist_path))
        self.imgB_hist.setScaledContents(True)

        # imgD, imgD_path = self.gray_D(img1)
        # imgD = np.abs(imgA - imgB)
        imgD = imgA - imgB
        self.imgD_show.setPixmap(self.show_image(imgD))
        # self.imgD_show.setPixmap(QPixmap(imgD_path))
        self.imgD_show.setScaledContents(True)
        imgD_hist_path = ip.plot_hist(imgD, file_path, 'D')
        self.imgD_hist.setPixmap(QPixmap(imgD_hist_path))
        self.imgD_hist.setScaledContents(True)

    # Average Grayscale
    def gray_A(self, imgA):
        # return (imgA[:, :, 0] + imgA[:, :, 1] + imgA[:, :, 2])/3
        return imgA[:, :, 0]/3 + imgA[:, :, 1]/3 + imgA[:, :, 2]/3

    # Luma Grayscale
    def gray_B(self, imgB):
        return 0.299*imgB[:, :, 0] + 0.587*imgB[:, :, 1] + 0.114*imgB[:, :, 2]

    # def gray_D(self, img):
    #     imgD = np.zeros(np.shape(img))
    #     img_delta = self.gray_A(img) - self.gray_B(img)
    #     imgD[:, :, 0] = img[:, :, 0]*(1/3-0.299)*255
    #     imgD[:, :, 1] = img[:, :, 1]*(0.587-1/3)*255
    #     imgD[:, :, 2] = img[:, :, 2]*(1/3-0.114)*255
    #     save_path = self.img1_path.split(
    #         '/')[-1].split('.')[0] + '_gray_delta.png'
    #     cv2.imwrite(save_path, imgD)
    #     return img[:, :, 0]*(1/3-0.299)*255+img[:, :, 1]*(0.587-1/3)*255+img[:, :, 2]*(1/3-0.114)*255, save_path

    # Open Image2
    def img2_open(self):
        file_path = QFileDialog.getOpenFileName(self, "Folder")[0]
        self.img2_path = file_path
        if not file_path.endswith('.jpg') or file_path.endswith('.png') or file_path.endswith('.BMP'):
            return
        self.img2 = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), 1)
        self.img2_show.setPixmap(QPixmap(file_path))
        self.img2_show.setScaledContents(True)
        self.img2_botton.adjustSize()
        self.img2_switch = {0: self.gray_B(
            self.img2), 1: self.gray_B(self.img2)}

        self.imgG_show.setPixmap(self.show_image(
            self.img2_switch[self.comboBox.currentIndex()]))
        self.imgG_show.setScaledContents(True)
        imgG_hist_path = ip.plot_hist(
            self.img2_switch[self.comboBox.currentIndex()], file_path, 'G')
        self.imgG_hist.setPixmap(QPixmap(imgG_hist_path))
        self.imgG_hist.setScaledContents(True)

    def show_image(self, array):
        Qimage = QImage(np.uint8(
            array), array.shape[1], array.shape[0], array.shape[0], QImage.Format_Grayscale8)
        Qimage = QPixmap(Qimage)
        return Qimage

    def show_ouput(self):
        self.output_arr[self.output_arr > 255] = 255
        self.output_arr[self.output_arr < 0] = 0
        self.imgO_show.setPixmap(self.show_image(self.output_arr))
        self.imgO_show.setScaledContents(True)
        output_hist_path = ip.plot_hist(self.output_arr)
        self.imgO_hist.setPixmap(QPixmap(output_hist_path))
        self.imgO_hist.setScaledContents(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
