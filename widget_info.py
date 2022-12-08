# -*- coding: utf-8 -*-

"""
Author: Fallen
Email: sigernmenz_he@henu.edu.cn
Date: 2022/3/3
Desc: 信息板块，包括FPS,目标检测的结果等
"""

from PySide2.QtWidgets import (QGroupBox, QLabel, QWidget, QVBoxLayout)
import random

class WidgetInfo(QWidget):
    def __init__(self):
        super(WidgetInfo, self).__init__()

        vbox = QVBoxLayout()

        """
        板块的确立
        数量，FPS，以及(X,Y,Z,DIS)
        """

        self.number = QLabel('NO. ')
        self.label_fps = QLabel('FPS: ')
        self.label_x = QLabel('Coordinate x: ')
        self.label_y = QLabel('Coordinate y: ')
        self.label_z = QLabel('Coordinate z: ')
        self.label_dis = QLabel('Coordinate dis: ')
        vbox.addWidget(self.label_fps)
        vbox.addWidget(self.label_x)
        vbox.addWidget(self.label_y)
        vbox.addWidget(self.label_z)
        vbox.addWidget(self.label_dis)

        """
        Here is the re-opened section for energy saving and emission reduction, 
        including functions such as current measurement and current regulation.
        CM = 电流监测 and CR = 电流调节
        Note: Please cancel it in computer design
        """
        # begin
        # self.CM = QLabel('CM: ')
        # self.CR = QLabel('CR: ')
        # self.state = QLabel('State: ')
        # self.area = QLabel('Area: ')
        #
        # vbox.addWidget(self.CM)
        # vbox.addWidget(self.CR)
        # vbox.addWidget(self.state)
        # vbox.addWidget(self.area)
        # end

        box = QGroupBox()
        box.setLayout(vbox)

        _vbox = QVBoxLayout()
        _vbox.setContentsMargins(0, 0, 0, 0)
        _vbox.addWidget(box)
        self.setLayout(_vbox)

    def update_fps(self, fps):
        self.label_fps.setText(f'FPS: { "" if fps <= 0 else round(fps, 1)}')

    def update_3D(self, text_x, text_y, text_z, text_dis):
        self.label_x.setText(f'Coordinate x: {"" if text_x == 0 else round(text_x, 1)}')
        self.label_y.setText(f'Coordinate y: {"" if text_y == 0 else round(text_y, 1)}')
        self.label_z.setText(f'Coordinate z: {"" if text_z == 0 else round(text_z, 1)}')
        self.label_dis.setText(f'Coordinate dis: {"" if text_dis <= 0 else round(text_dis, 1)}')

    # This is a test function, no used in any item.
    def add_new_item(self, len_obj):
        vbox =[]
        for i in range(len_obj):
            vbox.append(QVBoxLayout())

            self.number = QLabel('NO. ')
            self.label_fps = QLabel('FPS: ')
            self.label_x = QLabel('Coordinate x: ')
            self.label_y = QLabel('Coordinate y: ')
            self.label_z = QLabel('Coordinate z: ')
            self.label_dis = QLabel('Coordinate dis: ')
            vbox[i].addWidget(self.number)
            vbox[i].addWidget(self.label_fps)
            vbox[i].addWidget(self.label_x)
            vbox[i].addWidget(self.label_y)
            vbox[i].addWidget(self.label_z)
            vbox[i].addWidget(self.label_dis)

            box = QGroupBox()
            box.setLayout(vbox)

            _vbox = QVBoxLayout()
            _vbox.setContentsMargins(0, 0, 0, 0)
            _vbox.addWidget(box)
            self.setLayout(_vbox)

    # 1 sun light IS 35mA, square = 0.21cm2, old_time
    def add_current(self):
        cm = 35.0
        cm = cm + random.random()
        cr = 1000 + random.random()*10
        state = 1
        area = 0.21 + random.random()*0.1
        self.CM.setText(f'CM: {"" if cm <= 0 else round(cm, 1)}')
        self.CR.setText(f'CR: {"" if cr <= 0 else round(cr, 1)}')
        self.state.setText(f'State: {"" if state <= 0 else "Working"}')
        self.area.setText(f'Area: {"" if area <= 0 else round(area, 2)}')
