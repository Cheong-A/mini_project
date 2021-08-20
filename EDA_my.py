
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')
import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline 주피터 노트북에 옮길때만

sns.set(font_scale=2)
#데이터 파일 불러오기
data = pd.read_csv('D:\\mini_project\\data\\train.csv')
