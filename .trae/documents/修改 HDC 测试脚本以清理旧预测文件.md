1. 修改 `clean_old_files` 函数，添加 `*_predicted.json` 到清理列表中
2. 在 Step 4 开始前调用 `clean_old_files` 函数，清理旧的预测文件
3. 确保清理操作只针对预测相关文件，不影响训练数据

修改位置：
- 函数 `clean_old_files`：添加 `*_predicted.json` 到 patterns 列表
- Step 4 开始前：调用 `clean_old_files([TRAIN_DATA_DIR, TEST_DATA_DIR])` 清理旧文件

这样可以确保每次预测前都会删除旧的预测结果，使用最新的模型生成结果。