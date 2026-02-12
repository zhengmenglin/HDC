from config import config

print('当前阈值配置:')
print('predict_threshold:', config.get('test.predict_threshold'))
print('predict_margin:', config.get('test.predict_margin'))
