# 代码模板

## 环境设置

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# 显示设置
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)
```

## 数据读取与清洗

```python
# 读取数据
df = pd.read_csv('data.csv')  # 或 pd.read_excel('data.xlsx')

# 大文件分块读取
chunks = pd.read_csv('large_file.csv', chunksize=100000)
df = pd.concat(chunks)

# 基本信息
print(f"维度: {df.shape}")
print(f"\n数据类型:\n{df.dtypes}")
print(f"\n缺失值:\n{df.isnull().sum()}")
print(f"\n重复行: {df.duplicated().sum()}")

# 缺失值处理
df['col'].fillna(df['col'].median(), inplace=True)  # 用中位数填充
df.dropna(subset=['important_col'], inplace=True)   # 删除关键列缺失行

# 异常值处理（IQR方法）
Q1, Q3 = df['col'].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
df_clean = df[(df['col'] >= lower) & (df['col'] <= upper)]
```

## 可视化模板

### 单变量分布

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 直方图
axes[0].hist(df['col'], bins=30, edgecolor='black', alpha=0.7)
axes[0].set_title('直方图')

# 箱线图
axes[1].boxplot(df['col'])
axes[1].set_title('箱线图')

# 密度图
df['col'].plot(kind='kde', ax=axes[2])
axes[2].set_title('密度图')

plt.tight_layout()
plt.savefig('distribution.png', dpi=150, bbox_inches='tight')
```

### 分组箱线图（最常用）

```python
fig, ax = plt.subplots(figsize=(10, 6))

# notch显示中位数置信区间
df.boxplot(column='value', by='group', ax=ax, notch=True)

# 或用seaborn（更美观）
sns.boxplot(x='group', y='value', data=df, notch=True)
plt.title('分组对比')
plt.savefig('boxplot_comparison.png', dpi=150, bbox_inches='tight')
```

### 散点图与相关性

```python
# 散点图矩阵
sns.pairplot(df[['var1', 'var2', 'var3']], diag_kind='kde')
plt.savefig('pairplot.png', dpi=150, bbox_inches='tight')

# 相关系数热力图
corr_matrix = df[numeric_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
plt.title('相关系数热力图')
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
```

## 统计检验

```python
# 独立样本t检验
group1 = df[df['group'] == 'A']['value']
group2 = df[df['group'] == 'B']['value']
t_stat, p_value = stats.ttest_ind(group1, group2)
print(f"t统计量: {t_stat:.4f}, p值: {p_value:.4f}")

# 效应量 (Cohen's d)
cohens_d = (group1.mean() - group2.mean()) / np.sqrt(
    ((len(group1)-1)*group1.std()**2 + (len(group2)-1)*group2.std()**2) /
    (len(group1) + len(group2) - 2)
)
print(f"Cohen's d: {cohens_d:.4f}")

# 卡方检验
contingency_table = pd.crosstab(df['cat1'], df['cat2'])
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
print(f"卡方统计量: {chi2:.4f}, p值: {p_value:.4f}")
```

## 机器学习建模

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 分类模型
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# 特征重要性
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

# 回归模型
reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
```

## 时间序列

```python
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# 时序分解
result = seasonal_decompose(df['value'], model='additive', period=12)
result.plot()
plt.savefig('seasonal_decompose.png', dpi=150, bbox_inches='tight')

# ARIMA预测
model = ARIMA(df['value'], order=(1, 1, 1))
fitted = model.fit()
forecast = fitted.forecast(steps=12)
```
