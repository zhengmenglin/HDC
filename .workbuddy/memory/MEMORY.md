# D:/HDC 项目长期笔记（精简版 / 细节见 docs）

## 定位与目标
「超声波探伤 HDC 系统」：超维计算做钢轨伤损**点级分类**（滑窗中心点，9 通道独立建模）。
不是 D:/fpga 硬件测试项目。论文目标（2026-10-13）= **HDC vs 视觉识别对照**，主线：
数据准备（60 万库检索 + 弱标签收割）→ HDC 点级识别 vs 视觉基线 → 对照实验 → 论文。
框架稿：`docs/项目数据流与总体框架分析_20261013.md`。切分纪律：**按录制分组**（步进重叠 888px）。

## 硬约束
- 通道（1-based）：俯仰 1-6=70°、7-8=37°、9=0°；水平 1-4=α(12~20°)、5-9=0°；
  拼孔值 10,60,20,50,30,40,0,70,80。⚠ config.py 的 channels 是 **0-based**（[8]=物理通道9=俯仰0°）。
- CSV：`*_hdc.csv`=encoding+channel_0-8；`*_label.csv`=encoding+label_0-8；
  `*_predicted_label.csv`=predicted_label_*。encoding = original_e − BeamOffset[ch]。
- 标签：0=正常，1-11=伤损，12-19=正常结构（接头/螺孔/导线孔/焊缝/鱼鳞/轨底锈蚀/焊筋/道岔）。
- 指标只算二值（>0 为正），见 hdc_utils.calculate_metrics。

## 环境
- 主力解释器：`C:\Users\...\Programs\Python\Python312\python.exe`（3.12.8）；托管 3.13 无依赖，勿用。
- 项目 venv：`D:\HDC\.venv`（已补 opencv-headless 4.10 + openpyxl；**numpy 1.26.4 / torch 2.2.0+cu118 勿升**）。
- 双击 .py 走 py 启动器默认 3.14（无库）⇒ 用桌面 `启动图案检索工具.pyw`（转交 venv pythonw）。
- GUI 启动：`cd /d D:\HDC` 后 `…Python312\python.exe main.py`。

## 代码骨架
`hdc_point_level_classifier.py`（DIM=8192, NUM_LEVELS=256, WINDOW=128, CENTER_IDX=63, NUM_TYPES=20；
编码 = Level(值) ⊕ bind(Random通道) ⊕ bind(Random位置) → sum → normalize）；
`test_hdc_unified.py`（唯一测试入口）；`hdc_utils.py`；`ultrasonic_data_converter.py`；`gui/`；
`docs/软件详细设计.md`（最权威）。

## HDC 现状（根因已定位，细节见 docs/算法原理图解与问题清单_20261012.md）
2026-02-14 P=R=F1=0 / Total Pred=0。四层叠加：
1) `_create_encoder` 全项目无定义 ⇒ 9 通道 + 已有 .pth 时 load 必崩（定时炸弹）；
2) train 仅 1 文件、正样本只在通道 8（8 个点），通道 0-7 正样本=0；
3) Centroid 零初始化 + add_online "is_wrong 全 False 即 return" ⇒ 原型恒零（2/180 非零）；
4) 判据 `md > ns + margin` 在原型范数失衡时恒假（ns 0.918 vs md 0.184）。
🔴 铁律：**编码与原型必须同源**（错配会让判据退化成噪声触发器，实测 1288 个预测全误报）。
其他缺陷：首训 OnlineHD / load 用 NeuralHD；增量学习 learned_files 恒空；指标混算 train+test；
`process_prediction_results` 用全局 predictions 填单文件矩阵（跨文件串台）；Level 256 挤在 1 span
（相邻 cos 0.9961，超出 8192 维分辨力）；randomness 不是可用旋钮（r=1 悬崖崩）。
`config.json` 不存在 ⇒ 默认 threshold=0 / margin=0 / channels=[0..8] / epochs=1 / onlinehd。

## 关键数据事实
BeamAngle ⇒ 通道 0-5=70°、6-7=37°、8=0°。channel_8 非零率 73.9%（底波连续），其余 2~4.3%。
标签：train `{8:8}`；test/180123 `{6:46,7:52,8:61}` ⇒ 通道 6/7 训练零正样本必漏。
`Total True=93` 只统计 label_8（9 通道全统计为 191）⇒ 改 channels 会静默换分母，历史指标不可比。

## 工具链（详见各文件头）
- `tools/find_pattern.py` + `test_find_pattern.py`（22 项回归）：彩色标记提取 → 位移投票 → 残差 NCC。
- `tools/find_pattern_batch.py`：批量检索（`run_batch()` 供 GUI 复用），输出 XLSX/CSV + 复核图。
- `tools/find_pattern_ui.py` + `build_single_ui.py` → `tools/dist/图案检索工具.py`（单文件分发）。
- `tools/scale_search.py`：大规模分层检索（T1 录制键→T2 日期+机号→T3 日期→T4 全库）。
  真数据 311 查询 × 63.47 万（shengyu）+ 2.4 万（第三批，平铺）。`--only-missed` 增量复检；
  `prebuild` 逐组落盘可续跑。**合并终版：是 263 / 疑似 9 / 否 39**。
- `tools/check_viewer.py`：命中检查器（三视图 / 翻页 / 批注断点续查），读 `<out>/检索结果.csv`。
- `tools/px2encoding.py`：`x_px → K=x∓N → E=K−D → encoding → CSV 行`；calibrate/convert/label。
  已标定 200922_17387（M,D=−464374）、210201_18277（M,D=−9085）；pilot 样本池 1855/1645。
  产出 `output/dataset_prep/`（清单 xlsx + 标定表 + 标签 CSV）。

## 踩过的坑（硬结论）
- B 显 1888×512，285.67 px/m，0 点 x=945；标记调色板必须含 **cyan (0,192,240)**；纯红 (240,0,0)
  与基线同色，无法提取（少量查询受限）。
- **dmap 带屏蔽必须与标记框保护配对**：y350-412 屏蔽红基线/红字会误杀轨底伤损标记；且全零模板
  会让 matchTemplate 输出常量 1.0 伪装完美匹配 ⇒ `dmap(a, protect=marks)` + `ncc_map` 退化守卫。
- 密集标记（焊缝/鱼鳞/锈蚀）覆盖率被低估 ⇒ **二次判定通道**（ncc≥0.75 且 cov≥0.5）。
- 单标记查询 + 重复场景有"泛型噪声底"（同组多文件 0.7~0.86 并列）；dmap 残差近乎色盲
  （绿标/紫标同分）⇒ 兜底全扫须"单峰突出 top1−top2≥0.15 + 颜色校验"。
- **步进方向分两种**：0646 系 `K=x+N`，第三批多为 `K=x−N`；单步内等价，必须用锚点孪生命中判 mode。
- 标定三坑：同构接头重复（210201 有 14+ 个 6/6 解）→ 锚点窗口 + 裁片绿标记交叉验证；蓝标记有
  ~18px 系统偏移 → 只用绿标记。
- ProcessPoolExecutor 脚本必须 `if __name__ == "__main__"`（Windows spawn）。
- GUI 脚本收尾 `os._exit`；测 GUI 别用 msys `timeout` 包装（假"卡死"）；输出写文件别用 `| tail`。
- 同文件多处 Edit 必须串行，改完 grep/语法复核（本机实测会静默丢改动）。

## 规模化卡点
项目内**无 .xxc→JSON 导出器**（现 JSON 出自设备侧软件）；数据就绪"片段"仅 8 个，
其余 110 个命中录制待导出。60 万库实测：建索引 351ms/图（12 进程 ≈2h/10 万张）、
检索 402s/311 查询；GPU 颜色分类 9.9ms/图、1D 剖面相关 60 万≈3s/查询（RTX 3060）。

## 外部资料
`docs/超维计算2026进展调研.md`：Frontiers in AI 10.3389/frai.2026.1690492（相关表征/正交）、
arXiv 2609.38471（level 码本去随机化）、EWSHM 2026 10.58286/33696（SHM 损伤 HDC，结构最像）、
arXiv 2205.07920（level 算法出处）、THDC arXiv 2602.00116（可训练嵌入，维度 10000→64）。
