# BeamOffset调整修复计划

## 问题分析

根据文档要求：
- `encoding`: 经过BeamOffset调整的编码值 (`original_e - beam_offset[channel]`)
- 需要从原始JSON中读取BeamOffset配置
- 对每个通道的编码值进行调整

## 实施步骤

### 步骤1：修改data_adapter.py
- 添加BeamOffset解析功能
- 在转换数据时应用beam_offset调整
- 保存原始编码值和调整后的编码值

### 步骤2：修改test_channel8_only.py
- 在生成1_hdc.json时，根据beam_offset计算original_e
- 使用公式：original_e = encoding + beam_offset[channel]

### 步骤3：更新数据格式
- 1.csv: 调整后的encoding + channel_0-8
- 1_label.csv: 调整后的encoding + label_0-8
- 1_predicted_label.csv: 调整后的encoding + predicted_label_0-8
- 1_hdc.json: 使用original_e（encoding + beam_offset）生成

## BeamOffset配置格式

从JSON中读取：
```json
{
  "BeamOffset": {
    "0": -4,    // 通道0的beam_offset
    "1": 98,    // 通道1的beam_offset
    ...
    "8": 257     // 通道8的beam_offset
  }
}
```

## 调整公式

- 调整后的encoding: `adjusted_encoding = original_e - beam_offset[channel]`
- 原始encoding: `original_e = adjusted_encoding + beam_offset[channel]`

## 预期结果

1. 成功解析BeamOffset配置
2. 在数据转换时应用beam_offset调整
3. 在生成1_hdc.json时恢复原始encoding值
4. 所有CSV文件都使用调整后的encoding