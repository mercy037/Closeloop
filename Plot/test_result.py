import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# 显示正负号与中文不显示问题
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('darkgrid', {'font.sans-serif':['SimHei', 'Arial']})

# 去除部分warning
import warnings
warnings.filterwarnings('ignore')

plt.figure(dpi=150)
L = [3,2,1,0,4]
sns.boxplot(L)
plt.show()
