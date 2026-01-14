import pandas as pd
from pathlib import Path

test_dir = Path('d:\\HDC\\data\\test')

for file in test_dir.glob('*.csv'):
    if 'hdc' in file.name:
        df = pd.read_csv(file)
        label_counts = df['label'].value_counts().to_dict()
        print(f'{file.name}: {label_counts}')
