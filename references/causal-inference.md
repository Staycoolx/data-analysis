# 因果推断方法详解

## 核心思想

### 因果图 (Pearl学派)
用DAG可视化变量间因果关系，区分：
- **混杂因子**：同时影响X和Y，需控制
- **中介因子**：X→M→Y路径上，视研究目的决定是否控制
- **对撞因子**：X→C←Y，不应控制（控制会引入偏差）

### 潜在结果模型 (Rubin学派)
- 反事实框架：每个个体有两个潜在结果 Y(1) 和 Y(0)
- 因果效应 = Y(1) - Y(0)
- 核心问题：同一个体只能观察到一个结果

## 方法详解

### 1. 倾向得分匹配 (PSM)

**思路**：为处理组找特征相似的对照组样本

**步骤**：
```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

# 1. 估计倾向得分
ps_model = LogisticRegression()
ps_model.fit(X_covariates, treatment)
propensity_scores = ps_model.predict_proba(X_covariates)[:, 1]

# 2. 匹配（最近邻）
treated_idx = np.where(treatment == 1)[0]
control_idx = np.where(treatment == 0)[0]

nn = NearestNeighbors(n_neighbors=1)
nn.fit(propensity_scores[control_idx].reshape(-1, 1))
distances, indices = nn.kneighbors(propensity_scores[treated_idx].reshape(-1, 1))

# 3. 计算ATT
matched_control_idx = control_idx[indices.flatten()]
att = outcome[treated_idx].mean() - outcome[matched_control_idx].mean()
```

**适用场景**：搜索/推荐/广告的前置分析

### 2. 双重差分法 (DID)

**思路**：(处理组后-处理组前) - (对照组后-对照组前)

**关键假设**：平行趋势假设（干预前两组趋势一致）

```python
import statsmodels.formula.api as smf

# DID回归
# Y = β0 + β1*treated + β2*post + β3*(treated*post) + ε
# β3 即为DID估计量

model = smf.ols('outcome ~ treated + post + treated:post', data=df).fit()
did_effect = model.params['treated:post']
```

**适用场景**：政策评估、策略全量后的长期效果

### 3. 断点回归 (RDD)

**思路**：阈值附近个体几乎随机分配

**类型**：
- Sharp RDD：阈值完全决定处理状态
- Fuzzy RDD：阈值影响处理概率

```python
from rdrobust import rdrobust

# 运行变量centered在阈值
running_var_centered = running_var - threshold

# RDD估计
result = rdrobust(y=outcome, x=running_var_centered)
print(f"RDD效应: {result.Estimate['tau.cl']}")
```

**适用场景**：基于阈值的激励策略（如发文超过10000字的奖励）

### 4. 合成控制法

**思路**：用多个对照单位加权合成一个"虚拟对照组"

```python
from SparseSC import fit

# 需要面板数据：多个单位、多个时间点
# 处理前数据用于找权重，处理后用于估计效应
```

**适用场景**：单一处理单位的政策评估（如某城市政策）

**限制**：数据完备性要求高

### 5. 工具变量法 (IV)

**思路**：找一个只通过X影响Y的外生变量Z

**条件**：
- 相关性：Z与X相关
- 排他性：Z只通过X影响Y
- 外生性：Z与误差项不相关

```python
from linearmodels.iv import IV2SLS

# 两阶段最小二乘
model = IV2SLS.from_formula('outcome ~ 1 + [endogenous ~ instrument]', data=df)
result = model.fit()
```

**适用场景**：有合适工具变量时（实践中较少）

## 实操流程

1. **绘制因果图**：基于业务理解画出变量关系
2. **识别变量类型**：区分混杂/中介/对撞因子
3. **选择控制方法**：根据数据特点选择PSM/DID/RDD等
4. **检验假设**：验证方法所需的前提假设
5. **敏感性分析**：检验结果对假设违背的稳健性
