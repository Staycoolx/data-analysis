# 数据分析与报告提效助手 📊

🇺🇸 [English Guide](README.md)

这是一个为 Claude Code 设计的全栈数据分析与报告工具箱，将 **D-D-S-P 分析管道**（描述、诊断、细分、预测）与**多专家深度分析**和**专业 HTML/PPT 报告输出**整合为一体。

> **核心理念**：先思考，再执行。不仅仅是跑脚本，而是像分析师一样推理。完成任务后，主动指出用户可能没有注意到的问题、趋势或机会。

---

## 🌟 核心升级

本版本将两个分析工作流整合为一：

| 工作流 | 适用场景 | 输出格式 |
|--------|---------|---------|
| **D-D-S-P 管道** | 探索性分析、快速洞察 | 终端 / Markdown |
| **多专家深度分析** | 正式报告、数据 >500 行或字段 >10 个 | HTML 报告 / PPTX |

---

## 🔄 决策流程

```
接到任务
    ↓
需要读取 Excel/PPTX？→ 是 → scripts/read_excel.py / read_pptx.py
    ↓
Phase 1：D-D-S-P 分析管道（脚本驱动）
    ↓
需要正式报告输出？
    ├─ 否（快速探索）→ 终端 / Markdown 输出
    └─ 是（报告/深度分析）→ Phase 2：多专家并行分析 → Phase 3：HTML / PPT
```

---

## 🛠 分析脚本

### Phase 1：D-D-S-P 管道

#### 1. 全局体检 (Describe) — `auto_eda.py`
快速了解数据全貌（质量、分布、统计量）。
- 自动检测缺失值和异常值
- 生成分布直方图和相关性热力图
```bash
python3 scripts/auto_eda.py data.csv --output report_dir --target 目标列名
```

#### 2. 归因分析 (Diagnose) — `analyze_drivers_optimized.py`
找出影响目标变量的核心因子。
- 随机森林排列重要性 (Permutation Importance) 排名
- 自动识别分类任务 vs 回归任务
```bash
python3 scripts/analyze_drivers_optimized.py data.csv 目标列名 --output driver_report
```

#### 3. 分组洞察 (Segment) — `analyze_groups.py`
对比不同细分群体的表现。
- 自动对数值型变量（年龄、价格等）进行分箱
- 按组聚合关键指标（均值、总和、计数）
```bash
python3 scripts/analyze_groups.py data.csv 分组列名 目标列名 --agg mean sum count
```

#### 4. 预测建模 (Predict) — `predict_target.py`
预测未来结果或对新数据进行分类。
- 训练随机森林基准模型
- 输出含预测值的 CSV 和可复用的 `.joblib` 模型文件
```bash
python3 scripts/predict_target.py data.csv 目标列名 --output prediction_result
```

#### 5. 时序分析 — `forecast_timeseries_std.py`
分析长期趋势、季节性和假日效应。
- 分解趋势 (Trend) + 季节性 (Seasonal) 成分
- 分析小时趋势和周末/假日效应
```bash
python3 scripts/forecast_timeseries_std.py data.csv 数值列 --datetime_col 时间列
```

### 数据读取辅助

| 脚本 | 用途 |
|------|------|
| `read_excel.py` | 读取 Excel 文件 → markdown / CSV / JSON |
| `read_pptx.py` | 读取 PPTX 结构与内容 |

---

## 📊 Phase 2：多专家深度分析

当数据 >500 行、字段 >10 个，或用户要求正式报告时触发。

**四阶段工作流：**

1. **数据理解** — 数据维度、时间跨度、字段清单、初步洞察
2. **专家选角** — 选取 3-5 个互补的专家角色（定量/战略/风险/行为等视角）
3. **并行分析** — 每个专家作为独立的 subagent 同时运行
4. **统一综合** — 以「高级分析师」视角整合所有发现（最终报告不出现专家名字）

---

## 🎨 Phase 3：报告输出层

### HTML 报告（默认）

11 种风格随机选择（未指定时）：

**经典系**：Financial Times · McKinsey · The Economist · Goldman Sachs · Swiss/NZZ

**设计系**：Stamen Design · Fathom · Sagmeister & Walsh · Takram · Irma Boom · Build

风格色值/字体/布局参考 → `references/report-style-gallery.md`

**布局底线（必须遵守）：**
```css
body { max-width: 1200px; margin: 0 auto; padding: 40px 48px; }
```

### PPT 输出

```bash
node scripts/html2pptx.js slides.html output.pptx   # 单文件转换
node scripts/build_pptx.js                           # 多页合并
```

PPT 风格参考 → `references/visual-design-system.md`

---

## 🚀 快速开始

### 环境依赖
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels holidays joblib
node --version  # PPT 转换需要 Node.js
```

### 安装
```bash
git clone https://github.com/Staycoolx/data-analysis.git
```

---

## 📁 项目结构

```
data-analysis/
├── SKILL.md                          # Claude Code 技能指南
├── README.md                         # 英文文档
├── README_CN.md                      # 中文文档
├── scripts/
│   ├── auto_eda.py                   # D：全局体检
│   ├── analyze_drivers_optimized.py  # D：归因分析
│   ├── analyze_groups.py             # S：分组洞察
│   ├── predict_target.py             # P：预测建模
│   ├── forecast_timeseries_std.py    # 时序分析
│   ├── read_excel.py                 # Excel 读取
│   ├── read_pptx.py                  # PPTX 读取
│   ├── html2pptx.js                  # HTML → PPTX 转换
│   └── build_pptx.js                 # 多页 PPTX 合并
└── references/
    ├── causal-inference.md           # PSM / DID / RDD 方法
    ├── code-templates.md             # 可复用代码模板
    ├── report-style-gallery.md       # 11 种 HTML 报告风格
    ├── html-templates.md             # HTML 可视化组件库
    ├── visual-design-system.md       # PPT 设计系统
    ├── workflows.md                  # 详细工作流规范
    └── ad-analytics.md               # 投放/广告分析领域知识
```

## 📝 License
MIT License
