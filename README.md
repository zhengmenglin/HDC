# 钢轨探伤B显数据JSON解析与可视化工具

## 项目概述

该工具用于解析和可视化钢轨探伤B显数据，根据《钢轨探伤B显数据JSON数据交换格式》规范编写，支持多种JSON格式输入，生成直观的B显图像和数据摘要。

## 项目结构

```
HDC/
├── gui/                    # 上位机界面模块
│   ├── __init__.py
│   ├── main_window.py      # 主窗口
│   ├── point_selector.py   # 点选择器
│   ├── type_selector.py    # 类型选择器
│   └── annotation_manager.py # 标注管理器
├── core/                   # 核心功能模块
│   ├── __init__.py
│   └── parse_flaw_detection_json.py # 数据解析和可视化
├── utils/                  # 工具脚本模块
│   ├── __init__.py
│   ├── HDC.py             # HDC算法核心
│   ├── data_reader.py     # 数据读取工具
│   ├── generate_hex_data.py # HEX数据生成
│   └── hdc_*.py          # HDC相关实现
├── data/                   # 数据文件
│   ├── __init__.py
│   ├── *.json             # JSON数据文件
│   ├── *.hex              # HEX数据文件
│   ├── *.xlsx             # Excel数据文件
│   └── *.txt              # 标签文件
├── docs/                   # 文档文件
│   ├── __init__.py
│   ├── *.md               # Markdown文档
│   ├── *.pdf              # PDF文档
│   └── *.docx             # Word文档
├── torchhd/               # TorchHD库
├── main.py                # 主程序入口
└── README.md             # 项目说明
```

## 功能特性

### 数据解析功能
- 支持多种JSON格式：JSON Lines格式、JSON数组格式、单行JSON数组
- 解析B显数据：B1/B2点数据、系统参数、作业信息等
- 数据修正：深度值修正、横向移动距离修正、灵敏度过滤

### 可视化功能
- B显点绘制：根据通道分配不同颜色，根据幅度设置点大小
- 信息面板：作业信息、通道统计、系统参数等
- 交互功能：缩放、保存图像、实时更新

### 上位机功能
- 图形用户界面：基于PyQt5的现代化界面
- X轴滑动条：控制显示范围的实时拖动
- 数据标注：点选择、类型标注、标注保存
- 选中的点信息显示：最多显示5行，支持滚动

### 点选择功能
- 单选模式：选择单个点，再次点击取消选择
- 多选模式：按住Shift键可多选
- 实时显示：选中点信息实时更新

## 安装和运行

### 环境要求
- Python 3.6+
- PyQt5
- matplotlib
- numpy

### 安装依赖
```bash
pip install PyQt5 matplotlib numpy
```

### 运行程序
```bash
# 启动上位机界面
python main.py

# 命令行模式解析单个文件
python core/parse_flaw_detection_json.py data/螺孔.json

# 命令行模式解析并指定输出
python core/parse_flaw_detection_json.py data/螺孔.json output.png --no-show
```

## 使用说明

### 上位机界面使用
1. 启动程序：`python main.py`
2. 打开文件：点击"文件" → "打开"，选择JSON文件
3. 查看图像：在主界面查看B显图像和信息面板
4. 调整范围：使用底部的X轴滑动条控制显示范围
5. 数据标注：点击"标注" → "开始标注"，选择类型后在图像上选择点

### 命令行使用
```bash
python core/parse_flaw_detection_json.py <json文件路径> [输出图像文件] [--no-show]
```

## 文件格式

### 输入JSON格式
```json
{
  "E": 123,
  "B1": [0, 100, 80],
  "Sensitivity": [0, 30, 1, 35],
  "Bureau": "北京铁路局",
  "Operator": "张三"
}
```

### 标注数据JSON格式
```json
{
  "annotations": [
    {
      "id": "1",
      "type": "螺孔",
      "points": [
        {"E": 101, "B": "B1", "channel": 0, "depth": 100, "amplitude": 80}
      ]
    }
  ]
}
```

## 开发说明

### 模块说明
- `gui/`：上位机界面相关代码
- `core/`：核心数据解析和可视化功能
- `utils/`：各种工具脚本和算法实现
- `data/`：测试数据和示例文件
- `docs/`：项目文档和规范说明

### 代码规范
- 使用中文注释
- 遵循PEP8编码规范
- 使用类型注解
- 模块化设计，职责分离

## 许可证

本项目遵循相应许可证条款。