# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\Yun\Desktop\master\ImageProcessing\C1HW05-2022\C1HW05-2022\hw5_gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1052, 722)
        MainWindow.setAutoFillBackground(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.img2_text = QtWidgets.QTextBrowser(self.centralwidget)
        self.img2_text.setGeometry(QtCore.QRect(20, 370, 331, 291))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.img2_text.setFont(font)
        self.img2_text.setObjectName("img2_text")
        self.img2_0 = QtWidgets.QLabel(self.centralwidget)
        self.img2_0.setGeometry(QtCore.QRect(20, 370, 331, 291))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.img2_0.setFont(font)
        self.img2_0.setAlignment(QtCore.Qt.AlignCenter)
        self.img2_0.setObjectName("img2_0")
        self.img1_show = QtWidgets.QLabel(self.centralwidget)
        self.img1_show.setGeometry(QtCore.QRect(20, 40, 331, 291))
        self.img1_show.setText("")
        self.img1_show.setAlignment(QtCore.Qt.AlignCenter)
        self.img1_show.setObjectName("img1_show")
        self.img1_button = QtWidgets.QPushButton(self.centralwidget)
        self.img1_button.setGeometry(QtCore.QRect(20, 40, 331, 291))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.img1_button.setFont(font)
        self.img1_button.setObjectName("img1_button")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(700, 10, 331, 321))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.tabWidget.setFont(font)
        self.tabWidget.setAutoFillBackground(True)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.rgb_button = QtWidgets.QRadioButton(self.tab)
        self.rgb_button.setGeometry(QtCore.QRect(20, 20, 281, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.rgb_button.setFont(font)
        self.rgb_button.setChecked(True)
        self.rgb_button.setObjectName("rgb_button")
        self.cmy_button = QtWidgets.QRadioButton(self.tab)
        self.cmy_button.setGeometry(QtCore.QRect(20, 60, 281, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.cmy_button.setFont(font)
        self.cmy_button.setObjectName("cmy_button")
        self.hsi_button = QtWidgets.QRadioButton(self.tab)
        self.hsi_button.setGeometry(QtCore.QRect(20, 100, 281, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.hsi_button.setFont(font)
        self.hsi_button.setObjectName("hsi_button")
        self.xyz_button = QtWidgets.QRadioButton(self.tab)
        self.xyz_button.setGeometry(QtCore.QRect(20, 140, 281, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.xyz_button.setFont(font)
        self.xyz_button.setObjectName("xyz_button")
        self.lab_button = QtWidgets.QRadioButton(self.tab)
        self.lab_button.setGeometry(QtCore.QRect(20, 180, 281, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.lab_button.setFont(font)
        self.lab_button.setObjectName("lab_button")
        self.yuv_button = QtWidgets.QRadioButton(self.tab)
        self.yuv_button.setGeometry(QtCore.QRect(20, 220, 281, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.yuv_button.setFont(font)
        self.yuv_button.setObjectName("yuv_button")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.label_3 = QtWidgets.QLabel(self.tab_2)
        self.label_3.setGeometry(QtCore.QRect(20, 20, 81, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.tab_2)
        self.label_4.setGeometry(QtCore.QRect(20, 60, 81, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.progressBar = QtWidgets.QProgressBar(self.tab_2)
        self.progressBar.setGeometry(QtCore.QRect(20, 100, 281, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.progressBar.setFont(font)
        self.progressBar.setProperty("value", 100)
        self.progressBar.setTextVisible(False)
        self.progressBar.setObjectName("progressBar")
        self.color_slider = QtWidgets.QSlider(self.tab_2)
        self.color_slider.setGeometry(QtCore.QRect(20, 141, 281, 31))
        self.color_slider.setMinimum(2)
        self.color_slider.setMaximum(16)
        self.color_slider.setPageStep(1)
        self.color_slider.setProperty("value", 4)
        self.color_slider.setOrientation(QtCore.Qt.Horizontal)
        self.color_slider.setObjectName("color_slider")
        self.color1_button = QtWidgets.QPushButton(self.tab_2)
        self.color1_button.setGeometry(QtCore.QRect(100, 20, 111, 31))
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(10)
        font.setKerning(True)
        self.color1_button.setFont(font)
        self.color1_button.setObjectName("color1_button")
        self.color2_button = QtWidgets.QPushButton(self.tab_2)
        self.color2_button.setGeometry(QtCore.QRect(100, 60, 111, 31))
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(10)
        self.color2_button.setFont(font)
        self.color2_button.setObjectName("color2_button")
        self.pseudo_button = QtWidgets.QPushButton(self.tab_2)
        self.pseudo_button.setGeometry(QtCore.QRect(20, 180, 181, 31))
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(12)
        self.pseudo_button.setFont(font)
        self.pseudo_button.setObjectName("pseudo_button")
        self.color_lv = QtWidgets.QLabel(self.tab_2)
        self.color_lv.setGeometry(QtCore.QRect(220, 180, 81, 31))
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(12)
        self.color_lv.setFont(font)
        self.color_lv.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.color_lv.setObjectName("color_lv")
        self.ps_label1 = QtWidgets.QLabel(self.tab_2)
        self.ps_label1.setGeometry(QtCore.QRect(30, 220, 101, 31))
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(12)
        self.ps_label1.setFont(font)
        self.ps_label1.setText("")
        self.ps_label1.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.ps_label1.setObjectName("ps_label1")
        self.ps_label2 = QtWidgets.QLabel(self.tab_2)
        self.ps_label2.setGeometry(QtCore.QRect(130, 220, 171, 31))
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(12)
        self.ps_label2.setFont(font)
        self.ps_label2.setText("")
        self.ps_label2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.ps_label2.setObjectName("ps_label2")
        self.label_3.raise_()
        self.label_4.raise_()
        self.progressBar.raise_()
        self.color_slider.raise_()
        self.color1_button.raise_()
        self.color2_button.raise_()
        self.pseudo_button.raise_()
        self.color_lv.raise_()
        self.ps_label2.raise_()
        self.ps_label1.raise_()
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.seg_label2 = QtWidgets.QLabel(self.tab_3)
        self.seg_label2.setGeometry(QtCore.QRect(130, 100, 171, 31))
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(12)
        self.seg_label2.setFont(font)
        self.seg_label2.setText("")
        self.seg_label2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.seg_label2.setObjectName("seg_label2")
        self.seg_label1 = QtWidgets.QLabel(self.tab_3)
        self.seg_label1.setGeometry(QtCore.QRect(30, 100, 101, 31))
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(12)
        self.seg_label1.setFont(font)
        self.seg_label1.setText("")
        self.seg_label1.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.seg_label1.setObjectName("seg_label1")
        self.k = QtWidgets.QLabel(self.tab_3)
        self.k.setGeometry(QtCore.QRect(220, 60, 81, 31))
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(12)
        self.k.setFont(font)
        self.k.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.k.setObjectName("k")
        self.k_slider = QtWidgets.QSlider(self.tab_3)
        self.k_slider.setGeometry(QtCore.QRect(20, 21, 281, 31))
        self.k_slider.setMinimum(2)
        self.k_slider.setMaximum(16)
        self.k_slider.setPageStep(1)
        self.k_slider.setProperty("value", 8)
        self.k_slider.setOrientation(QtCore.Qt.Horizontal)
        self.k_slider.setObjectName("k_slider")
        self.seg_button = QtWidgets.QPushButton(self.tab_3)
        self.seg_button.setGeometry(QtCore.QRect(20, 60, 211, 31))
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(12)
        self.seg_button.setFont(font)
        self.seg_button.setObjectName("seg_button")
        self.tabWidget.addTab(self.tab_3, "")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 10, 331, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setAutoFillBackground(False)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_0 = QtWidgets.QLabel(self.centralwidget)
        self.label_0.setGeometry(QtCore.QRect(20, 340, 331, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.label_0.setFont(font)
        self.label_0.setAlignment(QtCore.Qt.AlignCenter)
        self.label_0.setObjectName("label_0")
        self.label_size = QtWidgets.QLabel(self.centralwidget)
        self.label_size.setGeometry(QtCore.QRect(240, 21, 111, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.label_size.setFont(font)
        self.label_size.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_size.setObjectName("label_size")
        self.img2_show = QtWidgets.QLabel(self.centralwidget)
        self.img2_show.setGeometry(QtCore.QRect(360, 40, 331, 291))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.img2_show.setFont(font)
        self.img2_show.setAlignment(QtCore.Qt.AlignCenter)
        self.img2_show.setObjectName("img2_show")
        self.img2_text_2 = QtWidgets.QTextBrowser(self.centralwidget)
        self.img2_text_2.setGeometry(QtCore.QRect(360, 40, 331, 291))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.img2_text_2.setFont(font)
        self.img2_text_2.setObjectName("img2_text_2")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(360, 10, 331, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.label_5.setFont(font)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.img2_1 = QtWidgets.QLabel(self.centralwidget)
        self.img2_1.setGeometry(QtCore.QRect(360, 370, 331, 291))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.img2_1.setFont(font)
        self.img2_1.setAlignment(QtCore.Qt.AlignCenter)
        self.img2_1.setObjectName("img2_1")
        self.img2_text_3 = QtWidgets.QTextBrowser(self.centralwidget)
        self.img2_text_3.setGeometry(QtCore.QRect(360, 370, 331, 291))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.img2_text_3.setFont(font)
        self.img2_text_3.setObjectName("img2_text_3")
        self.label_1 = QtWidgets.QLabel(self.centralwidget)
        self.label_1.setGeometry(QtCore.QRect(360, 340, 331, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.label_1.setFont(font)
        self.label_1.setAlignment(QtCore.Qt.AlignCenter)
        self.label_1.setObjectName("label_1")
        self.img2_2 = QtWidgets.QLabel(self.centralwidget)
        self.img2_2.setGeometry(QtCore.QRect(700, 370, 331, 291))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.img2_2.setFont(font)
        self.img2_2.setAlignment(QtCore.Qt.AlignCenter)
        self.img2_2.setObjectName("img2_2")
        self.img2_text_4 = QtWidgets.QTextBrowser(self.centralwidget)
        self.img2_text_4.setGeometry(QtCore.QRect(700, 370, 331, 291))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.img2_text_4.setFont(font)
        self.img2_text_4.setObjectName("img2_text_4")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(700, 340, 331, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.img2_text_5 = QtWidgets.QTextBrowser(self.centralwidget)
        self.img2_text_5.setGeometry(QtCore.QRect(20, 40, 331, 291))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.img2_text_5.setFont(font)
        self.img2_text_5.setObjectName("img2_text_5")
        self.img2_text_5.raise_()
        self.img2_text_3.raise_()
        self.img2_text_2.raise_()
        self.img2_text_4.raise_()
        self.img2_text.raise_()
        self.img2_0.raise_()
        self.img1_show.raise_()
        self.img1_button.raise_()
        self.tabWidget.raise_()
        self.label.raise_()
        self.label_0.raise_()
        self.label_size.raise_()
        self.img2_show.raise_()
        self.label_5.raise_()
        self.img2_1.raise_()
        self.label_1.raise_()
        self.img2_2.raise_()
        self.label_2.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1052, 25))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionClose = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.actionClose.setFont(font)
        self.actionClose.setObjectName("actionClose")
        self.actionOpen_img = QtWidgets.QAction(MainWindow)
        self.actionOpen_img.setObjectName("actionOpen_img")
        self.menuFile.addAction(self.actionOpen_img)
        self.menuFile.addAction(self.actionClose)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.img2_text.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Arial\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.img2_0.setText(_translate("MainWindow", "Ouput array[0]"))
        self.img1_button.setText(_translate("MainWindow", "Open image"))
        self.rgb_button.setText(_translate("MainWindow", "RGB"))
        self.cmy_button.setText(_translate("MainWindow", "CMY"))
        self.hsi_button.setText(_translate("MainWindow", "HSI"))
        self.xyz_button.setText(_translate("MainWindow", "XYZ"))
        self.lab_button.setText(_translate("MainWindow", "L*a*b*"))
        self.yuv_button.setText(_translate("MainWindow", "YUV"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Part 1"))
        self.label_3.setText(_translate("MainWindow", "Color 1:"))
        self.label_4.setText(_translate("MainWindow", "Color 2:"))
        self.color1_button.setText(_translate("MainWindow", "Choose"))
        self.color2_button.setText(_translate("MainWindow", "Choose"))
        self.pseudo_button.setText(_translate("MainWindow", "Pseudo Color"))
        self.color_lv.setText(_translate("MainWindow", "lv. 4"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Part 2"))
        self.k.setText(_translate("MainWindow", "k = 8"))
        self.seg_button.setText(_translate("MainWindow", "Color Segmentation"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Part 3"))
        self.label.setText(_translate("MainWindow", "Input image"))
        self.label_0.setText(_translate("MainWindow", "Ouput array[0]"))
        self.label_size.setText(_translate("MainWindow", "size"))
        self.img2_show.setText(_translate("MainWindow", "Output image"))
        self.img2_text_2.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Arial\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.label_5.setText(_translate("MainWindow", "Ouput image"))
        self.img2_1.setText(_translate("MainWindow", "Ouput array[1]"))
        self.img2_text_3.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Arial\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.label_1.setText(_translate("MainWindow", "Ouput array[1]"))
        self.img2_2.setText(_translate("MainWindow", "Ouput array[2]"))
        self.img2_text_4.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Arial\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "Ouput array[2]"))
        self.img2_text_5.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Arial\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionClose.setText(_translate("MainWindow", "Close"))
        self.actionClose.setShortcut(_translate("MainWindow", "Ctrl+W"))
        self.actionOpen_img.setText(_translate("MainWindow", "Open img"))
        self.actionOpen_img.setShortcut(_translate("MainWindow", "Ctrl+Q"))
