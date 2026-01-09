1. **添加BeamWay字段解析**：在\_parse\_event方法中添加对'BeamWay'字段的解析，将射向信息存储到self.params\['beam\_way']字典中
2. **修改后向通道判断逻辑**：根据BeamWay值确定哪些是后向通道，而不是使用硬编码的通道号
3. **更新channel\_config字典**：根据JSON文档中的通道顺序和射向信息更新通道配置
4. **修改BeamAngle解析**：在解析BeamAngle时，根据BeamWay值判断通道是否为后向通道，然后决定角度的正负
5. **添加通道映射关系**：明确A-I通道与0-8索引的对应关系

修改范围：

* FlawDetectionDataParser类的\_parse\_event方法

* channel\_config字典的定义

* BeamAngle字段的解析逻辑

具体实现：

1. 在\_parse\_event方法中添加BeamWay字段解析
2. 根据BeamWay值和JSON文档注释确定后向通道
3. 修改BeamAngle解析，使用BeamWay值判断后向通道
4. 更新channel\_config字典，确保通道顺序和射向信息正确
5. 添加清晰的注释，说明通道映射关系

