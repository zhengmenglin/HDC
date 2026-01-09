#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
类型选择对话框
用于选择伤损类型或结构类型进行标注
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QRadioButton,
    QButtonGroup, QPushButton, QTabWidget, QWidget
)
from PyQt5.QtCore import Qt


class TypeSelectorDialog(QDialog):
    """类型选择对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择标注类型")
        self.setGeometry(300, 300, 400, 400)
        
        # 初始化选择结果
        self.selected_type = None
        self.selected_subtype = None
        
        # 创建布局
        self.main_layout = QVBoxLayout(self)
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)
        
        # 创建伤损类型标签页
        self.create_damage_type_tab()
        
        # 创建结构类型标签页
        self.create_structure_type_tab()
        
        # 创建按钮
        self.create_buttons()
    
    def create_damage_type_tab(self):
        """创建伤损类型标签页"""
        damage_tab = QWidget()
        damage_layout = QVBoxLayout(damage_tab)
        
        # 伤损类型列表
        self.damage_types = {
            "核伤类": {
                "1": "轨头外侧核伤",
                "2": "轨头中心核伤",
                "3": "轨头内侧核伤",
                "4": "轨头内、外、中心核伤",
                "5": "轨头内、中心核伤",
                "6": "轨头外、中心核伤"
            },
            "水平裂纹类": {
                "7": "轨头水平分裂",
                "8": "轨头、轨腰分裂",
                "9": "轨腰水平裂"
            },
            "轨底裂纹类": {
                "10": "轨底断裂"
            },
            "螺孔裂纹类": {
                "11": "螺孔裂纹"
            },
            "其他类": {
                "12": "其他伤损"
            }
        }
        
        # 创建伤损类型按钮组
        self.damage_button_groups = {}
        
        for type_name, subtypes in self.damage_types.items():
            # 创建类型标题
            type_label = QLabel(f"<b>{type_name}</b>")
            damage_layout.addWidget(type_label)
            
            # 创建按钮组
            button_group = QButtonGroup(self)
            self.damage_button_groups[type_name] = button_group
            
            # 创建子类型单选按钮
            for subtype_id, subtype_name in subtypes.items():
                radio_button = QRadioButton(f"{subtype_id}. {subtype_name}")
                radio_button.setProperty("type_name", type_name)
                radio_button.setProperty("subtype_id", subtype_id)
                radio_button.setProperty("subtype_name", subtype_name)
                button_group.addButton(radio_button)
                damage_layout.addWidget(radio_button)
            
            # 添加分隔线
            damage_layout.addSpacing(10)
        
        self.tab_widget.addTab(damage_tab, "伤损类型")
    
    def create_structure_type_tab(self):
        """创建结构类型标签页"""
        structure_tab = QWidget()
        structure_layout = QVBoxLayout(structure_tab)
        
        # 结构类型列表
        self.structure_types = {
            "1": "接头",
            "2": "螺孔",
            "3": "导线孔",
            "4": "焊缝",
            "5": "鱼鳞伤",
            "6": "轨底锈蚀",
            "7": "轨底焊筋",
            "8": "道岔",
            "9": "其他"
        }
        
        # 创建结构类型按钮组
        self.structure_button_group = QButtonGroup(self)
        
        # 创建结构类型单选按钮
        for subtype_id, subtype_name in self.structure_types.items():
            radio_button = QRadioButton(f"{subtype_id}. {subtype_name}")
            radio_button.setProperty("subtype_id", subtype_id)
            radio_button.setProperty("subtype_name", subtype_name)
            self.structure_button_group.addButton(radio_button)
            structure_layout.addWidget(radio_button)
        
        self.tab_widget.addTab(structure_tab, "结构类型")
    
    def create_buttons(self):
        """创建按钮"""
        button_layout = QHBoxLayout()
        
        # 确定按钮
        self.ok_button = QPushButton("确定")
        self.ok_button.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_button)
        
        # 取消按钮
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        self.main_layout.addLayout(button_layout)
    
    def accept(self):
        """接受选择"""
        current_tab = self.tab_widget.currentIndex()
        
        if current_tab == 0:  # 伤损类型
            # 检查哪个按钮被选中
            for type_name, button_group in self.damage_button_groups.items():
                checked_button = button_group.checkedButton()
                if checked_button:
                    self.selected_type = "伤损"
                    self.selected_subtype = {
                        "id": checked_button.property("subtype_id"),
                        "name": checked_button.property("subtype_name"),
                        "category": type_name
                    }
                    break
        
        elif current_tab == 1:  # 结构类型
            checked_button = self.structure_button_group.checkedButton()
            if checked_button:
                self.selected_type = "结构"
                self.selected_subtype = {
                    "id": checked_button.property("subtype_id"),
                    "name": checked_button.property("subtype_name")
                }
        
        # 如果没有选择任何类型，显示错误信息
        if self.selected_type is None:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "警告", "请选择一个类型")
            return
        
        # 调用父类的accept方法
        super().accept()
    
    def get_selected_type(self):
        """获取选择结果"""
        return {
            "type": self.selected_type,
            "subtype": self.selected_subtype
        }
