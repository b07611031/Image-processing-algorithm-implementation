from msilib.schema import ComboBox
import sys
import numpy as np
import cv2
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import Ui_hw3_gui as ui
import img_processing as ip


class MainWindow(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.img1_button.clicked.connect(self.img1_open)

        self.box_button.toggled.connect(self.comboBox_filter)
        self.custom_button.toggled.connect(self.custom)
        self.sobel_button.toggled.connect(self.sobel)
        self.gaussian_button.toggled.connect(self.gaussian)
        self.log_button.toggled.connect(self.log)
        self.heq_button.toggled.connect(self.heq)
        self.local_button.toggled.connect(self.local)

        # self.box_size.valueChanged.connect(self.box_changed)
        self.custom_size.valueChanged.connect(self.custom_table_changed)

        self.actionQuit.triggered.connect(lambda: self.close())

    def rbutton_isChecked(self):
        if self.tabWidget.currentIndex() == 0:
            if self.box_button.isChecked():
                self.comboBox_filter()
            if self.custom_button.isChecked():
                self.custom()
            if self.gaussian_button.isChecked():
                self.gaussian()
        elif self.tabWidget.currentIndex() == 1:
            if self.sobel_button.isChecked():
                self.sobel()
            if self.log_button.isChecked():
                self.log()
        elif self.tabWidget.currentIndex() == 2:
            if self.heq_button.isChecked():
                self.heq()
            if self.local_button.isChecked():
                self.local()

    # Open Image1
    def img1_open(self):
        self.file_path = QFileDialog.getOpenFileName(
            self, "Open image file", '', '*.bmp;*.jpg;*.JPG;*.png')[0]
        if self.file_path == '':
            return
        # print(file_path)
        img = cv2.imdecode(np.fromfile(self.file_path, dtype=np.uint8), 1)
        # read and display in grayscale
        self.img_arr, img_luma_path = ip.gray_luma(img, self.file_path)
        self.img1_show.setPixmap(QPixmap(img_luma_path))
        self.img1_show.setScaledContents(True)
        self.img1_button.adjustSize()
        img_hist_path = ip.plot_hist(self.img_arr, img_luma_path, '')
        self.img1_hist.setPixmap(QPixmap(img_hist_path))
        self.img1_hist.setScaledContents(True)

        self.rbutton_isChecked()

    def comboBox_filter(self):
        if self.box_button.isChecked():
            # print('box_filter')
            kernel_name = str(self.comboBox.currentText())
            print(kernel_name)
            v = self.box_size.value()
            if v % 2 == 0:
                v -= 1
            self.box_size.setValue(v)
            if kernel_name == 'Box filter':
                output_arr, output_path = ip.box_filter(
                    self.img_arr, v, self.file_path)
            elif kernel_name == 'Median filter':
                output_arr, output_path = ip.median_filter(
                    self.img_arr, v, self.file_path)
            elif kernel_name == 'Max filter':
                output_arr, output_path = ip.max_filter(
                    self.img_arr, v, self.file_path)
            elif kernel_name == 'Min filter':
                output_arr, output_path = ip.min_filter(
                    self.img_arr, v, self.file_path)
            self.show_ouput(output_arr, output_path)

    def sobel(self):
        if self.sobel_button.isChecked():
            print('Sobel filter')
            output_arr, output_path = ip.sobel_filter(
                self.img_arr, self.file_path)
            self.show_ouput(output_arr, output_path)

    def gaussian(self):
        if self.gaussian_button.isChecked():
            print('Gaussian_filter')
            v = self.gaussian_size.value()
            if v % 2 == 0:
                v -= 1
            self.gaussian_size.setValue(v)
            sigma = self.gaussian_sigma.value()
            k = self.gaussian_K.value()
            output_arr, output_path = ip.gaussian_filter(
                self.img_arr, v, sigma, k, self.file_path)
            self.show_ouput(output_arr, output_path)

    def log(self):
        if self.log_button.isChecked():
            print('LoG_filter')
            v = self.log_size.value()
            if v % 2 == 0:
                v -= 1
            self.log_size.setValue(v)
            sigma = self.log_sigma.value()
            threshold = self.log_zero.value()
            output_arr, output_path = ip.log_filter(
                self.img_arr, v, sigma, threshold, self.file_path)
            self.show_ouput(output_arr, output_path)

    def heq(self):
        if self.heq_button.isChecked():
            print('Histogram_Equalization')
            output_arr, output_path = ip.hist_equalize(
                self.img_arr, self.file_path)
            self.show_ouput(output_arr, output_path)

    def local(self):
        if self.local_button.isChecked():
            print('Local_Enhancement')
            v = self.local_size.value()
            if v % 2 == 0:
                v -= 1
            self.local_size.setValue(v)
            k0 = self.k0.value()
            k1 = self.k1.value()
            k2 = self.k2.value()
            k3 = self.k3.value()
            cc = self.cc.value()
            output_arr, output_path = ip.local_enhancement(
                self.img_arr, v, k0, k1, k2, k3, cc, self.file_path)
            self.show_ouput(output_arr, output_path)

    def custom(self):
        if self.custom_button.isChecked():
            print('custom_filter')
            self.custom_filter()

    def custom_table_changed(self, v):
        if v % 2 == 0:
            v -= 1
        self.custom_size.setValue(v)
        self.custom_table.setColumnCount(v)
        self.custom_table.setRowCount(v)
        for c in range(v):
            self.custom_table.setColumnWidth(c, 30)

    def custom_filter(self):
        kernel = np.zeros((self.custom_size.value(), self.custom_size.value()))
        for i in range(self.custom_size.value()):
            for j in range(self.custom_size.value()):
                try:
                    kernel[i, j] = float(self.custom_table.item(i, j).text())
                except:
                    kernel[i, j] = 0.0
                    self.custom_table.setItem(i, j, QTableWidgetItem('0.'))
        print(kernel)
        output_arr, output_path = ip.convolution(
            self.img_arr, kernel, self.file_path, 'custom')
        self.show_ouput(output_arr, output_path)

    def show_ouput(self, img_arr, img_path):
        self.img2_show.setPixmap(QPixmap(img_path))
        self.img2_show.setScaledContents(True)
        img_hist_path = ip.plot_hist(img_arr, self.file_path, 'gray')
        self.img2_hist.setPixmap(QPixmap(img_hist_path))
        self.img2_hist.setScaledContents(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
