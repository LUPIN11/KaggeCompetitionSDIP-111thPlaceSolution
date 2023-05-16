import pandas as pd
import matplotlib.pyplot as plt

# 生成示例数据
df1 = pd.read_csv('vitOnValidSet.csv')
df2 = pd.read_csv('../PromptPredict/resultOnValidSet.csv')
# 绘制柱状图

plt.bar(range(len(df1)), df1['similarity'], width=1, alpha=0.5, label='vit')
plt.bar(range(len(df2)), df2['similarity'], width=1, alpha=0.5, label='clip')
print()
# 显示图表
plt.show()
