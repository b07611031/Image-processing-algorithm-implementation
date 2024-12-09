from msilib.schema import ComboBox
import sys
import numpy as np
import cv2
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import Ui_hw4_gui as ui
import img_processing as ip


class MainWindow(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setButtonGroup()
        self.img1_button.clicked.connect(self.img1_open)

        self.fft_button.toggled.connect(self.fft_inverse)
        self.fft_d_button.toggled.connect(self.fft_difference)
        self.ideal_button.toggled.connect(self.ideal)
        self.gaussian_button.toggled.connect(self.gaussian)
        self.butter_button.toggled.connect(self.butterworth)
        self.homo_button.toggled.connect(self.homomorphic)
        self.checkBox.clicked.connect(self.part4_trigger)
        self.motion_button.toggled.connect(self.motion_blur)
        self.unblur_button.toggled.connect(self.unblur)
        self.unblur_d_button.toggled.connect(self.unblur_difference)
        self.unblur_m_button.toggled.connect(self.motion_unblur)

        self.comboBox.activated.connect(self.part2_trigger)
        self.cutoff.valueChanged.connect(self.part2_trigger)
        self.order_n.valueChanged.connect(self.butterworth)
        self.gh.valueChanged.connect(self.homomorphic)
        self.gl.valueChanged.connect(self.homomorphic)
        self.d0.valueChanged.connect(self.homomorphic)
        self.noise_mean.valueChanged.connect(self.part4_trigger)
        self.noise_sigma.valueChanged.connect(self.part4_trigger)
        self.motion_a.valueChanged.connect(self.part4_trigger)
        self.motion_b.valueChanged.connect(self.part4_trigger)
        self.motion_t.valueChanged.connect(self.part4_trigger)
        self.comboBox_r.activated.connect(self.part4_trigger)
        self.cutoff_2.valueChanged.connect(self.unblur_trigger)

        self.actionClose.triggered.connect(lambda: self.close())
        self.actionOpen_img.triggered.connect(self.img1_open)

    def setButtonGroup(self):
        self.buttongroup = QButtonGroup(self)
        self.buttongroup.addButton(self.fft_button)
        self.buttongroup.addButton(self.fft_d_button)
        self.buttongroup.addButton(self.ideal_button)
        self.buttongroup.addButton(self.gaussian_button)
        self.buttongroup.addButton(self.butter_button)
        self.buttongroup.addButton(self.homo_button)
        self.buttongroup.addButton(self.motion_button)
        self.buttongroup.addButton(self.unblur_button)
        self.buttongroup.addButton(self.unblur_d_button)
        self.buttongroup.addButton(self.unblur_m_button)

    def rbutton_isChecked(self):
        self.fft_inverse()
        self.fft_difference()
        self.part2_trigger()
        self.homomorphic()
        self.part4_trigger()

    # Open Image1
    def img1_open(self):
        self.img_path = QFileDialog.getOpenFileName(
            self, "Open image file", '', '*.bmp;*.jpg;*.JPG;*.png')[0]
        if self.img_path == '':
            return
        # print(img_path)
        # read and display in grayscale
        img = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), 1)
        self.img_arr, img_luma_path = ip.gray_luma(img, self.img_path)
        self.img1_show.setPixmap(QPixmap(img_luma_path))
        self.img1_show.setScaledContents(True)
        self.img1_button.adjustSize()
        img_hist_path = ip.plot_hist(self.img_arr, img_luma_path, '')
        self.img1_hist.setPixmap(QPixmap(img_hist_path))
        self.img1_hist.setScaledContents(True)
        self.label_size.setText(f'({img.shape[0]}, {img.shape[1]})')
        self.fshift, spectrum_path, phase_path, fft_time = ip.fft(
            self.img_arr, self.img_path)
        self.label_fft_time.setText(f'{np.round(fft_time, 3)} s')
        self.img2_spectrum.setPixmap(QPixmap(spectrum_path))
        self.img2_spectrum.setScaledContents(True)
        self.img2_phase.setPixmap(QPixmap(phase_path))
        self.img2_phase.setScaledContents(True)
        self.rbutton_isChecked()

    def fft_inverse(self):
        if self.fft_button.isChecked():
            print('fft and ifft')
            output_arr, output_path = ip.ifft(
                self.fshift, self.img_path, save=True)
            self.show_ouput(output_arr, output_path)

    def fft_difference(self):
        if self.fft_d_button.isChecked():
            print('FFt difference')
            output_arr, output_path = ip.fft_d(
                self.img_arr, self.fshift, self.img_path)
            self.show_ouput(output_arr, output_path)

    def part2_trigger(self):
        self.ideal()
        self.gaussian()
        self.butterworth()

    def ideal(self):
        if self.ideal_button.isChecked():
            print('Ideal filter')
            output_arr, output_path = ip.ideal_filter(
                self.fshift, self.cutoff.value(), self.comboBox.currentText(), self.img_path)
            self.show_ouput(output_arr, output_path)

    def gaussian(self):
        if self.gaussian_button.isChecked():
            print('Gaussian filter')
            output_arr, output_path = ip.gaussian_filter(
                self.fshift, self.cutoff.value(), self.comboBox.currentText(), self.img_path)
            self.show_ouput(output_arr, output_path)

    def butterworth(self):
        if self.butter_button.isChecked():
            print('Butterworth filter')
            output_arr, output_path = ip.butterworth_filter(self.fshift, self.cutoff.value(
            ), self.order_n.value(), self.comboBox.currentText(), self.img_path)
            self.show_ouput(output_arr, output_path)

    def homomorphic(self):
        if self.homo_button.isChecked():
            print('Homomorphic filter')
            output_arr, output_path = ip.homo_filter(
                self.fshift, self.gh.value(), self.gl.value(), self.d0.value(), self.img_path)
            self.show_ouput(output_arr, output_path)

    def part4_trigger(self):
        self.motion_blur()
        self.unblur()
        self.unblur_difference()
        self.motion_unblur()

    def unblur_trigger(self):
        if self.comboBox_r.currentText() == 'Wiener filter':
            self.unblur()
            self.motion_unblur()
        self.unblur_difference()

    def motion_blur(self):
        if self.motion_button.isChecked():
            print('Motion blur')
            output_arr, output_path = ip.motion_filter(
                self.fshift, self.motion_a.value(), self.motion_b.value(), self.motion_t.value(), self.img_path)
            if self.checkBox.isChecked():
                print('Gaussian noise')
                output_arr, output_path = ip.noise_filter(
                    output_arr, self.noise_mean.value(), self.noise_sigma.value(), self.img_path)
            self.show_ouput(output_arr, output_path)

    def unblur(self):
        if self.unblur_button.isChecked():
            print('Unblur')
            gshift, hshift = ip.motion_filter(
                self.fshift, self.motion_a.value(), self.motion_b.value(), self.motion_t.value(), self.img_path, save=False)
            if self.checkBox.isChecked():
                print('Gaussian noise')
                gshift = ip.noise_filter(
                    gshift, self.noise_mean.value(), self.noise_sigma.value(), self.img_path, save=False)
            output_arr, output_path = ip.unblur_filter(gshift, self.comboBox_r.currentText(
            ), hshift, self.cutoff_2.value(), self.img_path)
            self.show_ouput(output_arr, output_path)

    def unblur_difference(self):
        if self.unblur_d_button.isChecked():
            print('Unblur_difference')
            gshift, hshift = ip.motion_filter(
                self.fshift, self.motion_a.value(), self.motion_b.value(), self.motion_t.value(), self.img_path, save=False)
            if self.checkBox.isChecked():
                print('Gaussian noise')
                gshift = ip.noise_filter(
                    gshift, self.noise_mean.value(), self.noise_sigma.value(), self.img_path, save=False)
            output_arr, output_path = ip.unblur_d_filter(
                gshift, hshift, self.cutoff_2.value(), self.img_path)
            self.show_ouput(output_arr, output_path)

    def motion_unblur(self):
        if self.unblur_m_button.isChecked():
            print('Motion unblur')
            output_arr, output_path = ip.motion_unblur_filter(self.fshift, self.motion_a.value(), self.motion_b.value(
                ), self.motion_t.value(), self.comboBox_r.currentText(), self.cutoff_2.value(), self.img_path)
            self.show_ouput(output_arr, output_path)

    def show_ouput(self, img_arr, img_path):
        self.img3_show.setPixmap(QPixmap(img_path))
        self.img3_show.setScaledContents(True)
        img_hist_path = ip.plot_hist(img_arr, self.img_path, 'gray')
        self.img3_hist.setPixmap(QPixmap(img_hist_path))
        self.img3_hist.setScaledContents(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


# delta(difference) : \u0394
# gamma : \u03B3
# sigma : \u03C3
