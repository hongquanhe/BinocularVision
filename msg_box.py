"""
@Author:Fallen
@File:msg_box.py
@Time:2022-04-11 19:14
"""
# coding=utf-8
from PySide2.QtGui import QIcon, QPixmap
from PySide2.QtWidgets import QMessageBox


class MsgSuccess(QMessageBox):
    """提示（操作成功）对话框，使用的时候设置提示信息即可"""

    def __init__(self):
        super(MsgSuccess, self).__init__()
        self.setWindowTitle('注意')
        self.setWindowIcon(QIcon('img/yologo.png'))
        pix = QPixmap('img/success.svg').scaled(48, 48)
        self.setIconPixmap(pix)
        self.setStyleSheet('font: 32px Microsoft YaHei')


class MsgWarning(QMessageBox):
    """注意对话框，使用的时候设置提示信息即可"""

    def __init__(self):
        super(MsgWarning, self).__init__()
        self.setWindowTitle('注意')
        self.setWindowIcon(QIcon('img/yologo.png'))
        pix = QPixmap('img/warn.svg').scaled(48, 48)
        self.setIconPixmap(pix)
        self.setStyleSheet('font: 16px Microsoft YaHei')
