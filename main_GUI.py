# coding:utf-8
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
import os
import cv2
import sys
import cv2
import struct
import numpy as np
from pose_onnx import Keypoint
import time

# ip='192.168.137.149'


model_path = r'model/s-pose.onnx'
# 实例化模型
keydet = Keypoint(model_path)


class FramerateCalculator:
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0

    def tick(self):
        """
        记录一帧的渲染，并计算帧率
        :return: 当前的帧率
        """
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        # 如果已经过去了一秒钟，我们重置计时器并计算帧率
        if elapsed_time >= 1.0:
            framerate = self.frame_count / elapsed_time
            self.start_time = current_time
            self.frame_count = 0
            return framerate
        return None


class Ui_MainWindow(object):

    def __init__(self):
        self.DeClassNameList = None
        self.imgPathList = ''  # 存储选择的图片路径
        self.DeClassNameList = ''  # 存储检测结果
        self.ConfigList = []  # 存储检测结果概率
        self.ret = None
        self.wendu = ''
        self.shidu = ''
        self.image = None
        self.tingzhi = True
        # 接入相机IP
        #self.camera_path = r"test_video/compose_video_1649207577312.mp4"
        #self.capture = cv2.VideoCapture(self.camera_path)
        #self.camera_path = r"ip"
        self.capture = cv2.VideoCapture(0)

    def setupUi(self, MainWindow):

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(900, 550)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.button_FangFa = QtWidgets.QPushButton(self.centralwidget)
        self.button_FangFa.setGeometry(QtCore.QRect(20, 400, 140, 50))
        self.button_FangFa.setObjectName("button_FangFa")

        self.label_Image = QtWidgets.QLabel(self.centralwidget)
        self.label_Image.setGeometry(QtCore.QRect(190, 30, 640, 350))
        self.label_Image.setStyleSheet("background-color: rgb(170, 170, 170);")
        self.label_Image.setText("")
        self.label_Image.setObjectName("label_Image")

        self.lable_Text = QtWidgets.QTextBrowser(self.centralwidget)
        self.lable_Text.setGeometry(QtCore.QRect(190, 400, 640, 100))
        # self.lable_Text.setStyleSheet("background-color: rgb(170, 255, 255);")
        self.lable_Text.setObjectName("lable_Text")

        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(10, 40, 141, 291))
        self.widget.setObjectName("widget")

        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")

        self.button_Open = QtWidgets.QPushButton(self.widget)
        self.button_Open.setObjectName("button_Open")
        self.verticalLayout.addWidget(self.button_Open)

        self.button_KaiShi = QtWidgets.QPushButton(self.widget)
        self.button_KaiShi.setObjectName("button_Kaishi")
        self.verticalLayout.addWidget(self.button_KaiShi)

        self.button_ShiShi = QtWidgets.QPushButton(self.widget)
        self.button_ShiShi.setObjectName("button_ShiShi")
        self.verticalLayout.addWidget(self.button_ShiShi)

        self.button_StopTime = QtWidgets.QPushButton(self.widget)
        self.button_StopTime.setObjectName("button_StopTime")
        self.verticalLayout.addWidget(self.button_StopTime)

        self.button_Cle = QtWidgets.QPushButton(self.widget)
        self.button_Cle.setObjectName("button_Cle")
        self.verticalLayout.addWidget(self.button_Cle)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1025, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # 监听按钮的单击
        self.button_Open.clicked.connect(self.DoOpen)  # 选择图片路径
        self.button_KaiShi.clicked.connect(self.DoKaiShi)  # 开始检测
        self.button_FangFa.clicked.connect(self.DoFangFa)  # 防治措施
        self.button_ShiShi.clicked.connect(self.DoShiShi)
        # self.timer = QTimer(MainWindow)
        # self.timer.timeout.connect(self.DoShiShi)  # 实时检测
        self.button_StopTime.clicked.connect(self.StopTime)  # 停止检测
        self.button_Cle.clicked.connect(self.Cle)  # 清空

        # self.imageNameList = []


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Posture  Safety Detection"))
        self.button_FangFa.setText(_translate("MainWindow", "Result"))
        self.button_Open.setText(_translate("MainWindow", "Open Path"))
        self.button_KaiShi.setText(_translate("MainWindow", "Start Detection"))
        self.button_ShiShi.setText(_translate("MainWindow", "Real-time Detection"))
        self.button_StopTime.setText(_translate("MainWindow", "Stop Detection"))
        self.button_Cle.setText(_translate("MainWindow", "Clear"))

    # 单击打开目录，选择图片
    def DoOpen(self):
        self.lable_Text.setText('')
        # 获取图片的路径
        imgPath = list(QtWidgets.QFileDialog.getOpenFileName(None, "select folder", ''))
        # print(imgPath[0])
        self.imgPathList = imgPath[0]
        self.label_Image.setScaledContents(True)  # 让图片自适应label大小
        # 绘制图片，显示出来
        self.pix = QtGui.QPixmap(imgPath[0])
        self.label_Image.setPixmap(self.pix)

    # 单击开始检测按钮，开始检测
    def DoKaiShi(self):
        global keydet
        path = self.imgPathList
        if path:
            image = cv2.imread(path)
            #image = cv2.imread(0)
            img = keydet.inference(image)
            self.label_Image.setScaledContents(True)  # 让图片自适应label大小
            self.label_Image.setPixmap(self.show(img))
            self.lable_Text.clear()
            self.DeClassNameList = "无"
            self.ConfigList = "无"
            self.ClassName()
        else:
            self.lable_Text.append('Please open the path and selet Image to detect！')


    # 单击防治方法按钮，展示防治方法
    def DoFangFa(self):
        self.ClassName()

    #  打开摄像头
    def startCamera(self):

        self.timer.start(1)
        self.lable_Text.clear()
        self.flag = 1

    # 实时检测
    def DoShiShi(self):
        global keydet
        framerate_calculator = FramerateCalculator()
        
        if self.ret is False:
            pass
        while True:
            self.ret, image = self.capture.read()
            
            img, action = keydet.inference(image)
            framerate = framerate_calculator.tick()
            if framerate is not None:
                print(f"Frame: {framerate:.2f} f/s")
            # cv2.imshow('keypoint', img)
            cv2.waitKey(1)
            self.label_Image.setScaledContents(True)  # 让图片自适应label大小
            # 绘制图片，显示出来
            self.label_Image.setPixmap(self.show(img))
            self.label_Image.setPixmap(self.show(img))
            self.lable_Text.clear()
            if action is not None:
                if action == "Normal":
                    self.DeClassNameList = "Savety"
                else:
                    self.DeClassNameList = "Dangerous"
            self.ClassName()

    # 停止检测
    def StopTime(self):
        self.tingzhi = False
        self.label_Image.clear()

    # 绘制图片
    def show(self, im):
        frame = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        return pixmap

    # 清空
    def Cle(self):
        self.label_Image.clear()
        self.lable_Text.clear()

    def ClassName(self):
        T_max = self.ConfigList.__str__()
        self.lable_Text.append('Detect Result:' + self.DeClassNameList)

# 主函数入口,打开UI界面
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())
