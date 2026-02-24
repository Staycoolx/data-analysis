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

---

## 六、平行趋势检验

### 6.1 为什么重要

DID 的核心假设是**平行趋势假设**（Parallel Trends Assumption）：
- 如果没有干预，处理组和对照组的结果趋势应该平行
- 这是 DID 有效性的关键前提

### 6.2 检验方法

#### 方法1：事件研究法（Event Study）

```python
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np

def event_study_did(df, treatment_col, outcome_col, time_col, group_col):
    """
    Event Study DID to test parallel trends and dynamic effects.
    """
    df = df.copy()
    df = df.sort_values(time_col)

    # Get unique time points
    unique_times = sorted(df[time_col].unique())
    n_times = len(unique_times)

    # Create time index
    time_map = {t: i for i, t in enumerate(unique_times)}
    df['time_idx'] = df[time_col].map(time_map)

    # Use middle period as reference (pre-treatment)
    ref_idx = n_times // 2
    df['relative_time'] = df['time_idx'] - ref_idx

    # Create treated indicator
    treated_val = df[treatment_col].unique()[0]
    df['treated'] = (df[treatment_col] == treated_val).astype(int)

    # Build event study formula
    relative_times = sorted(df['relative_time'].unique())
    event_terms = [f'event_{rt}' for rt in relative_times if rt != 0]
    treat_terms = [f'treated:event_{rt}' for rt in relative_times if rt != 0]

    # Create dummies
    for rt in relative_times:
        if rt != 0:
            df[f'event_{rt}'] = (df['relative_time'] == rt).astype(int)
            df[f'treated:event_{rt}'] = df['treated'] * df[f'event_{rt}']

    # Run regression
    formula = f'{outcome_col} ~ treated + ' + ' + '.join(event_terms) + ' + ' + ' + '.join(treat_terms)
    model = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df[group_col]})

    # Extract results
    results = []
    for rt in relative_times:
        if rt == 0:
            continue
        term = f'Treated:event_{rt}'
        if term in model.params:
            results.append({
                'relative_time': rt,
                'coef': model.params[term],
                'se': model.bse[term],
                'ci_low': model.conf_int().loc[term, 0],
                'ci_high': model.conf_int().loc[term, 1],
                'pvalue': model.pvalues[term]
            })

    return pd.DataFrame(results)

# Plot Event Study
def plot_event_study(event_results):
    """Plot event study with confidence intervals."""
    fig, ax = plt.subplots(figsize=(10, 6))

    times = event_results['relative_time'].values
    coefs = event_results['coef'].values
    ci_low = event_results['ci_low'].values
    ci_high = event_results['ci_high'].values

    ax.errorbar(times, coefs, yerr=[coefs - ci_low, ci_high - coefs],
                fmt='o-', capsize=4, linewidth=2, markersize=8,
                color='#2E86AB', ecolor='gray')

    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.axvline(x=0, color='green', linestyle='--', linewidth=1, alpha=0.5)

    # Shade pre-treatment period
    pre = event_results[event_results['relative_time'] < 0]
    if len(pre) > 0:
        ax.axvspan(pre['relative_time'].min() - 0.5, -0.5, alpha=0.1, color='blue')

    ax.set_xlabel('Relative Time (Periods from Treatment)')
    ax.set_ylabel('Treatment Effect')
    ax.set_title('Event Study: Dynamic Treatment Effects')
    plt.tight_layout()
    plt.show()

# Interpretation:
# - Pre-treatment coefficients should be near zero (parallel trends)
# - Post-treatment coefficients show dynamic treatment effects
```

#### 方法2：伪treatment检验

在干预前的时间点"假装"有干预，检验效应是否显著。

```python
def placebo_test(df, treatment_col, outcome_col, time_col, group_col, placebo_time_idx):
    """
    Placebo test: apply treatment at an earlier time point.
    """
    df = df.copy()
    unique_times = sorted(df[time_col].unique())

    # Create fake treatment at placebo time
    placebo_time = unique_times[placebo_time_idx]
    df['placebo_post'] = (df[time_col] >= placebo_time).astype(int)

    treated_val = df[treatment_col].unique()[0]
    df['treated'] = (df[treatment_col] == treated_val).astype(int)

    # Run DID with placebo treatment
    model = smf.ols(f'{outcome_col} ~ treated + placebo_post + treated:placebo_post',
                    data=df).fit(cov_type='cluster', cov_kwds={'groups': df[group_col]})

    placebo_effect = model.params.get('treated:placebo_post', 0)
    placebo_pvalue = model.pvalues.get('treated:placebo_post', 1)

    return {
        'placebo_effect': placebo_effect,
        'placebo_pvalue': placebo_pvalue,
        'is_valid': placebo_pvalue > 0.05  # Should NOT be significant
    }
```

### 6.3 平行趋势检验通过标准

| 检验方法 | 通过标准 |
|---------|---------|
| Event Study | 干预前所有系数接近0，置信区间包含0 |
| 伪treatment检验 | 干预前各时点效应均不显著(p>0.05) |
| 简单均值对比 | 两组干预前均值差异较小(<5%) |

---

## 七、稳健性检验

### 7.1 常用稳健性检验

| 检验方法 | 目的 |
|---------|-----|
| 不同控制组 | 更换对照组，检验结果是否一致 |
| 不同时间窗口 | 调整干预前后时间窗口长度 |
| 替换变量 | 用替代指标作为因变量 |
| 极端值剔除 | 剔除极端样本后重新估计 |
| 分样本回归 | 按不同群体分组检验 |

### 7.2 代码示例

```python
def robustness_checks(df, treatment_col, outcome_col, time_col, group_col):
    """
    Run multiple robustness checks.
    """
    results = {}

    # 1. Different control group
    # (Assume we have multiple groups, use different one as control)
    # ...

    # 2. Different time windows
    # ...

    # 3. Trim extreme values (top/bottom 1%)
    df_trimmed = df.copy()
    for col in [outcome_col]:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df_trimmed = df_trimmed[(df_trimmed[col] >= lower) & (df_trimmed[col] <= upper)]

    model = smf.ols(f'{outcome_col} ~ treated + post + treated:post',
                    data=df_trimmed).fit()
    results['trimmed'] = model.params['treated:post']

    return results
```

---

## 八、业务场景决策树

```
场景判断
    │
    ├─ 有明确的时间突变点？
    │   ├─ 是 → 有分组吗？
    │   │       ├─ 是 → DID（双重差分）
    │   │       └─ 否 → RDD（断点回归）
    │   │
    │   └─ 否 → 有分组但无时间突变？
    │           ├─ 是 → PSM（倾向得分匹配）
    │           └─ 否 → 需要重新设计分析方案
    │
    └─ 评估效果 vs 因果推断？
            ├─ 效果评估 → 简单比较即可
            └─ 因果推断 → 按上述决策树选择方法
```

### 常见业务场景对应方法

| 业务场景 | 推荐方法 |
|---------|---------|
| 推荐系统升级效果评估 | DID（不同渠道/版本对比） |
| 新功能上线效果 | DID / AB Test |
| 政策/补贴效果 | DID / 合成控制法 |
| 用户行为引导（如达到阈值奖励） | RDD |
| 渠道/人群策略效果 | PSM / DID |
| 会员权益价值 | PSM / 分组对比 |
