1. 在test\_accuracy函数中，在模型预测完成后添加CSV到HDC JSON转换步骤
2. 确保转换步骤同时处理测试数据和训练数据
3. 优化预测test数据的逻辑，添加错误处理，确保部分文件权限错误不会中断整个流程
4. 检查通道处理逻辑，确保通道值从原始JSON到最终hdc.json的正确性

修改文件：
d:\HDC\test\_full\_pipeline.py

主要修改点：

1. 在test\_accuracy函数的预测步骤后添加CSV到HDC JSON转换
2. 导入CsvToHdcJsonConverter
3. 为测试数据和训练数据分别执行转换
4. 确保通道处理一致，使用0-8通道

