## 修改内容

1. **修改文件**：d:\HDC\hdc_test.py
2. **修改位置**：第182行
3. **当前代码**：`test_files = glob.glob(os.path.join(test_data_dir, '*.xxc.json'))`
4. **修改后代码**：
   ```python
   # 获取所有.json文件
   all_json_files = glob.glob(os.path.join(test_data_dir, '*.json'))
   # 过滤掉标注文件（文件名包含"标注"的文件）
   test_files = [f for f in all_json_files if "标注" not in os.path.basename(f)]
   ```

## 修改理由

- 当前代码只测试.xxc.json文件，无法测试其他格式的JSON数据文件
- 用户要求测试所有非标注.json的.json文件
- 标注文件的命名格式通常包含"标注"关键词，如"170921_16229片段.xxc标注.json"
- 修改后可以处理各种类型的JSON数据文件，提高脚本的通用性

## 预期效果

- 脚本将测试指定目录下所有JSON文件，除了文件名包含"标注"的文件
- 保持原有测试逻辑不变，只修改文件过滤条件
- 提高脚本的灵活性和通用性，能够处理更多类型的测试数据

## 实现步骤

1. 使用glob获取所有.json文件
2. 使用列表推导式过滤掉标注文件
3. 保持原有处理逻辑不变