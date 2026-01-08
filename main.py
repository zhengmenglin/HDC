#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
钢轨探伤B显数据可视化工具主入口
"""

import sys
from PyQt5.QtWidgets import QApplication
from main_window import MainWindow


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())