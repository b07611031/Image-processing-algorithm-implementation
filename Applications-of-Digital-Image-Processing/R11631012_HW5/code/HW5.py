import sys
import numpy as np
import cv2
import time
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import Ui_hw5_gui as ui
import img_processing as ip


class MainWindow(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setButtonGroup()
        # self.img1_open()
        self.img1_button.clicked.connect(self.img1_open)

        self.rgb_button.toggled.connect(self.img2rgb)
        self.cmy_button.toggled.connect(self.img2cmy)
        self.hsi_button.toggled.connect(self.img2hsi)
        self.lab_button.toggled.connect(self.img2lab)
        self.xyz_button.toggled.connect(self.img2xyz)
        self.yuv_button.toggled.connect(self.img2yuv)

        self.color1_button.clicked.connect(self.color1_picker)
        self.color2_button.clicked.connect(self.color2_picker)
        self.pseudo_button.clicked.connect(self.img2pseudo)
        self.seg_button.clicked.connect(self.imgkmeans)
        self.color_slider.valueChanged.connect(self.color_sliding)
        self.k_slider.valueChanged.connect(self.k_sliding)

        # self.comboBox_r.activated.connect(self.part4_trigger)
        # self.cutoff_2.valueChanged.connect(self.unblur_trigger)

        self.actionClose.triggered.connect(lambda: self.close())
        self.actionOpen_img.triggered.connect(self.img1_open)

    def setButtonGroup(self):
        self.buttongroup = QButtonGroup(self)
        self.buttongroup.addButton(self.rgb_button)
        self.buttongroup.addButton(self.cmy_button)
        self.buttongroup.addButton(self.hsi_button)
        self.buttongroup.addButton(self.xyz_button)
        self.buttongroup.addButton(self.lab_button)
        self.buttongroup.addButton(self.yuv_button)
        self.color1_button.setStyleSheet(
            f"background-color:black; color:white; border-radius: 5px")
        self.color2_button.setStyleSheet(
            f"background-color:gray; color:black; border-radius: 5px")
        self.color1 = QColor(0, 0, 0)
        self.color2 = QColor(170, 170, 170)
        self.bar_gradient()

    def rbutton_isChecked(self):
        if self.tabWidget.currentIndex()==0:
            self.part1_trigger()

    # Open Image1
    def img1_open(self):
        # self.img_path = './HW05-Part 3-02.bmp'
        self.img_path = QFileDialog.getOpenFileName(
            self, "Open image file", '', '*.bmp;*.jpg;*.JPG;*.png')[0]
        if self.img_path == '':
            return
        self.img_arr = cv2.imdecode(
            np.fromfile(self.img_path, dtype=np.uint8), 1)
        self.img1_button.adjustSize()
        img_luma_path = ip.gray_luma(self.img_arr, self.img_path)
        self.img1_show.setPixmap(self.qmap_img(img_luma_path))
        self.rgb, output_path = ip.bgr2rgb(self.img_arr, self.img_path)
        self.label_size.setText(
            f'({self.img_arr.shape[0]}, {self.img_arr.shape[1]})')
        self.part1_show(output_path, 'RGB')
        self.img2_show.setPixmap(self.qmap_img(output_path))
        self.rbutton_isChecked()

    def img2rgb(self):
        if self.rgb_button.isChecked():
            print('RGB')
            output_path = ip.bgr2rgb(
                self.img_arr, self.img_path, save=True)
            self.part1_show(output_path, 'RGB')
            self.img2_show.setPixmap(self.qmap_img(output_path))

    def img2cmy(self):
        if self.cmy_button.isChecked():
            print('CMY')
            output_path = ip.rgb2cmy(self.rgb, self.img_path)
            self.part1_show(output_path, 'CMY')
            self.img2_show.setPixmap(self.qmap_img(output_path))

    def img2hsi(self):
        if self.hsi_button.isChecked():
            print('HSI')
            output_path = ip.rgb2hsi(self.rgb, self.img_path)
            self.part1_show(output_path, 'HSI')
            self.img2_show.setPixmap(self.qmap_img(output_path))

    def img2xyz(self):
        if self.xyz_button.isChecked():
            print('XYZ')
            output_path = ip.rgb2xyz(self.rgb, self.img_path)
            self.part1_show(output_path, 'XYZ')
            self.img2_show.setPixmap(self.qmap_img(output_path))

    def img2lab(self):
        if self.lab_button.isChecked():
            print('L*a*b*')
            output_path = ip.rgb2lab(self.rgb, self.img_path)
            self.part1_show(output_path, 'Lab')
            self.img2_show.setPixmap(self.qmap_img(output_path))

    def img2yuv(self):
        if self.yuv_button.isChecked():
            print('YUV')
            output_path = ip.rgb2yuv(self.rgb, self.img_path)
            self.part1_show(output_path, 'YUV')
            self.img2_show.setPixmap(self.qmap_img(output_path))

    def part1_trigger(self):
        self.img2rgb()
        self.img2cmy()
        self.img2hsi()
        self.img2xyz()
        self.img2lab()
        self.img2yuv()

    def part1_show(self, img_path, channel):
        self.img2_0.setPixmap(self.qmap_img(
            img_path.replace(channel, channel[0])))
        self.img2_1.setPixmap(self.qmap_img(
            img_path.replace(channel, channel[1])))
        self.img2_2.setPixmap(self.qmap_img(
            img_path.replace(channel, channel[2])))
        self.label_0.setText(channel[0])
        self.label_1.setText(channel[1])
        self.label_2.setText(channel[2])

    def color1_picker(self):
        self.color1 = QColorDialog.getColor()
        self.color1_button.setText('')
        self.color1_button.setStyleSheet(
            f"background-color:{self.color1.name()}; border:transparent; border-radius: 5px")
        self.bar_gradient()

    def color2_picker(self):
        self.color2 = QColorDialog.getColor()
        self.color2_button.setText('')
        self.color2_button.setStyleSheet(
            f"background-color:{self.color2.name()}; border:transparent; border-radius: 5px")
        self.bar_gradient()

    def bar_gradient(self):
        grad = f'QLinearGradient(x1:0, y1:0, x2:1, y2:0, stop:0 {self.color1.name()}, stop:1 {self.color2.name()})'
        self.progressBar.setStyleSheet("QProgressBar::chunk "
                                       "{"
                                       f"background: {grad};"
                                       "}")

    def img2pseudo(self):
        print('Pseudo')
        self.ps_label1.setText("Loading...")
        self.ps_label2.setText("")
        start_time = time.time()
        output_path = ip.pseudo(self.rgb, self.color_slider.value(), self.color1, self.color2, self.img_path)
        self.part1_show(output_path, 'RGB')
        self.ps_label1.setText("Done!")
        self.ps_label2.setText(f"Takes {round(time.time()-start_time, 3)}s.")
        self.img2_show.setPixmap(self.qmap_img(output_path))


    def color_sliding(self):
        _translate = QtCore.QCoreApplication.translate
        self.color_lv.setText(_translate(
            "MainWindow", f"lv. {self.color_slider.value()}"))
        
        
    def imgkmeans(self):
        print('Color Segmentation')
        self.seg_label1.setText("Loading...")
        self.seg_label2.setText("")
        start_time = time.time()
        output_path = ip.segmentation(self.rgb, self.k_slider.value(), self.img_path)
        self.part1_show(output_path, 'RGB')
        self.seg_label1.setText("Done!")
        self.seg_label2.setText(f"Takes {round(time.time()-start_time, 3)}s.")
        self.img2_show.setPixmap(self.qmap_img(output_path))


    def k_sliding(self):
        _translate = QtCore.QCoreApplication.translate
        self.k.setText(_translate(
            "MainWindow", f"lv. {self.k_slider.value()}"))
    

    def qmap_img(self, img_path):
        qmap = QPixmap(img_path)
        return qmap.scaled(self.img2_0.width(), self.img2_0.height(), QtCore.Qt.KeepAspectRatio)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


# delta(difference) : \u0394
# gamma : \u03B3
# sigma : \u03C3
