#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
点信息编辑对话框
用于修改选中点的深度和幅度信息
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QPushButton, QFormLayout, QMessageBox
)
from PyQt5.QtCore import Qt


class PointEditorDialog(QDialog):
    """点信息编辑对话框"""
    
    def __init__(self, point_info, parent=None):
        """
        初始化对话框
        
        Args:
            point_info: 点信息字典，包含 original 键，original 中包含 e, channel, depth, amplitude
            parent: 父窗口
        """
        super().__init__(parent)
        self.setWindowTitle("修改点信息")
        self.setGeometry(300, 300, 400, 200)
        self.point_info = point_info
        self.original_point = point_info['original']
        
        # 保存修改后的值
        self.modified_depth = self.original_point['depth']
        self.modified_amplitude = self.original_point['amplitude']
        
        # 创建布局
        self.layout = QVBoxLayout()
        
        # 创建表单布局
        form_layout = QFormLayout()
        
        # E值显示
        self.e_label = QLabel(f"{self.original_point['e']}")
        form_layout.addRow("E值:", self.e_label)
        
        # 通道显示
        self.channel_label = QLabel(f"{self.original_point['channel']}")
        form_layout.addRow("通道:", self.channel_label)
        
        # 深度输入
        self.depth_edit = QLineEdit(f"{self.original_point['depth']:.1f}")
        form_layout.addRow("深度:", self.depth_edit)
        
        # 幅度输入
        self.amplitude_edit = QLineEdit(f"{self.original_point['amplitude']:.1f}")
        form_layout.addRow("幅度:", self.amplitude_edit)
        
        # 添加表单布局到主布局
        self.layout.addLayout(form_layout)
        
        # 创建按钮布局
        button_layout = QHBoxLayout()
        
        # 确定按钮
        self.ok_button = QPushButton("确定")
        self.ok_button.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_button)
        
        # 取消按钮
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        # 添加按钮布局到主布局
        self.layout.addLayout(button_layout)
        
        # 设置布局
        self.setLayout(self.layout)
    
    def accept(self):
        """确定按钮点击事件"""
        # 验证输入
        try:
            depth = float(self.depth_edit.text())
            amplitude = float(self.amplitude_edit.text())
            
            if depth < 0:
                QMessageBox.warning(self, "警告", "深度不能为负数")
                return
            
            if amplitude < 0:
                QMessageBox.warning(self, "警告", "幅度不能为负数")
                return
            
            # 保存修改后的值
            self.modified_depth = depth
            self.modified_amplitude = amplitude
            
            # 调用父类的 accept 方法
            super().accept()
        except ValueError:
            QMessageBox.warning(self, "警告", "请输入有效的数字")
    
    def get_modified_values(self):
        """
        获取修改后的值
        
        Returns:
            tuple: (depth, amplitude) 修改后的深度和幅度值
        """
        return self.modified_depth, self.modified_amplitude
