---
name: data-analysis
description: |
  数据分析与报告提效全能助手。
  触发条件：
  - 用户需要分析数据、探索分布、发现模式或异常值
  - 用户需要数据可视化（直方图、箱线图、散点图、热力图等）
  - 用户需要统计建模、机器学习或预测分析
  - 用户需要因果推断（PSM、DID、RDD等）
  - 用户需要制作数据分析报告、HTML报告或PPT
  - 用户提到投放复盘、ROI测算、AB实验分析
  - 用户上传了 Excel/CSV 数据文件需要分析
---

# 数据分析与报告提效助手

## 指导原则

**先理解，后执行**——拿到任务先问「用户真正需要什么」。帮用户多想一步：完成后主动指出可能没注意到的问题、趋势或机会。

**数据诚实**——绝不编造数据，图表不误导（柱状图Y轴从0开始）。

---

## 一、分析流程决策

```
拿到数据
    ↓
是否需要 Excel/PPTX 读取？→ 是 → scripts/read_excel.py / read_pptx.py
    ↓
Phase 1: D-D-S-P 分析管道（脚本驱动）
    ↓
是否需要深度报告输出？
    ├─ 否（快速探索）→ 终端/Markdown 输出
    └─ 是（报告/深度分析）→ Phase 2: 多专家并行分析 → Phase 3: HTML/PPT 输出
```

**深度报告触发条件**（满足任一）：
- 用户明确要求「做报告」「深度分析」「出PPT」
- 数据 >500 行 或字段 >10 个
- 需要有受众的专业输出

---

## 二、Phase 1：D-D-S-P 分析管道

| 步骤 | 工具 | 命令 |
|------|------|------|
| **Describe** 看全貌 | `auto_eda.py` | `python3 scripts/auto_eda.py data.csv --output report_dir --target target_col` |
| **Diagnose** 找原因 | `analyze_drivers_optimized.py` | `python3 scripts/analyze_drivers_optimized.py data.csv target_col --output driver_report` |
| **Segment** 做细分 | `analyze_groups.py` | `python3 scripts/analyze_groups.py data.csv group_col target_col --agg mean sum count` |
| **Predict** 测未来 | `predict_target.py` | `python3 scripts/predict_target.py data.csv target_col --output prediction_result` |

**因果推断**：
| 工具 | 命令 |
|------|------|
| **DID分析** | `python3 scripts/analyze_did.py data.csv --treatment channel --outcome retention --time date --group user_id --output did_report` |

详见 [references/causal-inference.md](references/causal-inference.md) 和 [references/causal-scenarios.md](references/causal-scenarios.md)

**数据读取辅助**：
- Excel → `python3 scripts/read_excel.py file.xlsx`
- PPTX → `python3 scripts/read_pptx.py file.pptx`

---

## 三、Phase 2：多专家深度分析

面对有深度分析价值的数据集，采用「数据理解 → 专家选角 → 并行分析 → 统一呈现」四阶段流程。

### Phase 2.1 数据理解

读取数据后立即输出：
1. 数据维度（行×列）、时间跨度、字段清单
2. 基础统计（均值/中位数/极值/缺失率）
3. **初步洞察**（1-2 个立即可见的趋势或异常）

### Phase 2.2 专家选角（3-5个）

根据数据类型选取**视角互补**的专家角色。原则：
- 覆盖定量/定性/战略/风险/行为等不同维度
- 使用真实知名专家/机构名字增加代入感（如 Damodaran、McKinsey）
- 将角色陈述写入 md 文件供用户确认，再进入并行分析

### Phase 2.3 并行分析（subagent 架构）

每个专家角色使用独立 subagent（`run_in_background=true`）并行执行，每个 subagent prompt 包含：角色定义 + 数据路径 + 具体分析任务 + 输出格式要求。

### Phase 2.4 统一综合呈现

**关键原则：最终报告不出现任何专家角色名字。**

从「管理型高级分析师」视角融合所有专家结论：
- 按主题组织（非按角色）：基本面、风险、趋势、行为洞察
- 标题用结论句式（「CapEx翻倍，净现金首次转负」）而非描述句式（「资本支出分析」）

---

## 四、Phase 3：输出层

### 输出格式决策

| 用户意图 | 格式 |
|---------|------|
| 快速看数字、探索性分析 | 终端 + Markdown |
| 正式分析报告 | **HTML 报告**（默认） |
| 幻灯片/演示 | HTML → PPTX（`scripts/html2pptx.js`） |

### HTML 报告布局底线（必须遵守）

```css
html { background: [与body背景色一致]; }
body { max-width: 1200px; margin: 0 auto; padding: 40px 48px; }
```

### 报告风格（用户未指定时随机选择）

**经典系**：Financial Times / McKinsey / The Economist / Goldman Sachs / Swiss NZZ

**设计系**：Stamen Design / Fathom / Sagmeister & Walsh / Takram / Irma Boom / Build

风格色值/字体/布局参考 → `references/report-style-gallery.md`

### PPT 转换

```bash
node scripts/html2pptx.js slides.html output.pptx    # 单文件
node scripts/build_pptx.js                            # 多页合并
```

---

## 五、深度验证

### 不平衡数据处理

目标变量严重不平衡时（如欺诈 <1%）：

| 策略 | 方法 |
|------|------|
| 评估指标 | **PR-AUC**, F1-Score（禁用 Accuracy） |
| 数据层面 | SMOTE 过采样、Tomek Links 欠采样 |
| 模型层面 | `class_weight="balanced"` (sklearn) |

### 因果推断

详见 [references/causal-inference.md](references/causal-inference.md)

---

## 六、分析输出结构

```
核心结论（1-3句，管理层看这段就够了）
→ 数据支撑（具体数字、对比、趋势）
→ 异常/风险
→ 可执行建议（3-5条，按优先级）
→ 下一步（还能深挖什么）
```

---

## 七、注意事项

- **环境依赖**：默认用 `sklearn`，禁用 `xgboost/lightgbm/tabulate`，除非用户明确要求
- **并行安全**：sklearn 模型必须设 `n_jobs=1`
- **内存保护**：大数据集绘图前先抽样（`df.sample(2000)`）
- **中文显示**：`plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'sans-serif']`
- **审美禁区**：赛博霓虹 / 深蓝底 / 纯黑纯白底 / 金色在白底做文字
- **不确定时必须问**：数据字段含义不明、报告受众不明、涉及业务判断

## 参考文件索引

| 需要什么 | 文件 |
|---------|------|
| 报告风格参数（11种） | `references/report-style-gallery.md` |
| HTML 可视化组件库 | `references/html-templates.md` |
| PPT 风格参数 | `references/visual-design-system.md` |
| 详细工作流 | `references/workflows.md` |
| 投放/广告分析领域知识 | `references/ad-analytics.md` |
| 因果推断方法详解 | `references/causal-inference.md` |
| 因果推断业务场景库 | `references/causal-scenarios.md` |
| 代码模板 | `references/code-templates.md` |
