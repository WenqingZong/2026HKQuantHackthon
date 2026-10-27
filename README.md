# 2026 HK Quant Hackathon — ETF 因子与机器学习流水线

这是队伍里 **zwq 负责的模型部分**：从 Databento 拉原始行情 → 计算 Alpha101 因子 → 训练
MLP / XGBoost / LightGBM → 用 Rank IC 和 top-k 等权回测评估。

现在用的还是团队 ETF 池（`selected_etfs/`）加上 17 个只依赖日频 OHLCV 的 Alpha101 因子，
目的是**先把流水线跑通、把每个环节的接口定下来**。等因子清单定了，改一下
`config/default.yaml` 里的 `dataset.alphas` 就能直接正式训练，模型代码不用动。

## 目录结构

```
config/default.yaml       # 数据源、因子清单、切分日期、模型参数，全部在这里调
hkq/
  data.py                 # Databento 下载 + 宽表面板（Panel）
  alphas.py               # Alpha101 因子实现与注册表
  dataset.py              # 因子 + 标签 -> (date, symbol) 长表，时间切分
  models.py               # MLP / XGBoost / LightGBM，统一 fit / predict 接口
  evaluate.py             # Rank IC、绩效指标、top-k 等权回测
  pipeline.py             # 把上面几步串起来的下载 / 特征 / 训练流程
  config.py               # YAML 配置的解析
main.py                   # 命令行入口
tests/test_pipeline.py    # 因子公式、标签对齐、时间切分的校验
selected_etfs/            # 团队筛出来的训练池与交易池（含字段的完整表）
example_databento_data/   # 官方给的一份样例数据，可以对照字段
data/                     # 行情与因子数据集（git 忽略）
artifacts/                # 训练结果与报告（git 忽略）
```

## 快速开始

```powershell
uv sync                     # 装依赖（torch 走 pytorch-cu130 源，5090 可以直接用）
```

Databento 的 key 放在仓库根目录的 `.env` 里（这个文件不进 git）：

```
DATABENTO_API_KEY=db-xxxxxxxx
```

然后按顺序跑：

```powershell
uv run python main.py download   # 下载训练池 + 交易池的日频行情，约 10 秒、花费约 0.2 美元
uv run python main.py features   # 算因子、生成数据集，约 10 秒
uv run python main.py train      # 训练三个模型并评估，CPU/GPU 都不到 1 分钟
uv run python main.py all        # 上面三步一起跑
```

单独跑某个模型、换配置文件：

```powershell
uv run python main.py train --models xgboost --config config/default.yaml
uv run python -m pytest tests -q
```

## 数据

### 数据源怎么选的

比赛给的是 Databento 历史数据接口。日频策略只要 OHLCV，但不同数据集差别不小：

| 数据集 | 起始日期 | 覆盖 | 适合 |
| --- | --- | --- | --- |
| `EQUS.SUMMARY`（默认） | 2024-07-01 | 合并全市场成交，每个代码每天**一条**记录，成交量是全市场的 | 最干净，直接拿来用 |
| `EQUS.MINI` | 2023-03-28 | 合并行情但只含部分交易场所，成交量大约是真实值的 4% | 想要更长历史、且只用价格类因子 |
| `DBEQ.BASIC` | 2023-03-28 | 每个交易场所各一条记录，需要用成交量加权合并 | 需要更长历史 + 分场所明细 |

默认用 `EQUS.SUMMARY`：ETF 在十几个交易场所都有成交，只有合并后的成交量才是真实的，
而成交量类因子（alpha006 / alpha012 / alpha022 / alpha026 等）全指望它。代价是历史只有
约 570 个交易日，2024-07-01 之前的数据 Databento 没有合并版本。

想换数据集只改 `config/default.yaml` 的 `data.dataset` 和 `data.start`。
`hkq/data.py` 里已经处理了"同一交易日多条记录"的情况：开盘/收盘取成交量最大场所的价格，
最高/最低取各场所极值，成交量求和，所以换成 `DBEQ.BASIC` 也能直接跑。

### 下载了什么

- 代码池 = 训练池（`training_tickers.csv`，172 只）∪ 交易池（`trading_tickers.csv`，176 只）= **182 只**；
  训练用训练池，交易池先一起下下来备用。
- 字段：`date, symbol, open, high, low, close, volume`，落在 `data/raw/bars_daily.parquet`。
- 每次下载会把查询参数、行数、数据质量降级的日期写进 `bars_daily.manifest.json`。
  当前有 3 天是 Databento 标记的 degraded（2025-12-15、2026-01-12、2026-08-31），做归因时留意。
- 要分钟级数据，把 `data.schema` 改成 `ohlcv-1m` 即可，时间戳会自动换算成纽约时间再归日；
  但一年多的分钟数据量很大，先想清楚用不用得上。

## 因子

`hkq/alphas.py` 实现了 Alpha101（Kakushadze, 2015）里**只靠日频 OHLCV 就能算**的那一批：

```
alpha002  alpha006  alpha008  alpha009  alpha012  alpha014  alpha020  alpha022
alpha023  alpha026  alpha033  alpha034  alpha038  alpha040  alpha044  alpha053  alpha101
```

每个函数的 docstring 就是原文公式，可以直接和论文对照。刻意没实现的三类：

- 需要 **vwap** 的（如 alpha041 / alpha042 / alpha050）——日频 OHLCV 没有真实 vwap；
- 需要 **adv20 / 成交额** 的（如 alpha007 / alpha043）——虽然能用收盘价乘成交量凑，但会引入偏差；
- 需要 **行业分类** 的（如 alpha055 / alpha060）——现在只有 ETF 池，行业中性化留给后面。

另外 `alpha003`（`rank(open)` 与 `rank(volume)` 的时序相关）和 `alpha013`
（`rank(close)` 的协方差）在 ETF 池上是**退化**的：价格水平排序在 100 多只 ETF 之间几乎不随时间变化，
滚动相关系数直接是 NaN。这类"对价格水平做横截面排名"的因子换到 ETF 上都要重新想。

想加因子：在 `alphas.py` 里写一个`(Panel) -> DataFrame` 的函数，注册到 `ALPHAS`，
再把名字加进 `config/default.yaml` 的 `dataset.alphas`。

## 数据集

`hkq/dataset.py` 把宽表拼成 `(date, symbol)` 的长表，每一步都有明确的时间约束：

1. **标签**：`lag=1`、`horizon=5` —— 因子用当日收盘价算出来，最早只能下一个交易日成交，
   所以标签是"第 t+1 日进场、持有 5 个交易日"的收益。不做这个滞后等于自带前视偏差。
2. **标签标准化**：默认按日做横截面 `rank`。一是把回归变成排序，和"选前 k 只"的目标一致；
   二是去掉市场整体涨跌（beta），模型不用花容量去拟合"今天大盘涨不涨"。
   这一步是纯单调变换，不影响 Rank IC 的数值。
3. **因子标准化**：同样按日做横截面 `rank`，映射到 `[-0.5, 0.5]`，缺失值填 0（中性）。
   这样树模型和 MLP 都不用再套一层 scaler，也不会有跨期泄漏。
4. **预热期**：前 20 个交易日只用来把滚动窗口填满，不参与训练。
5. **切分**：train 2024-08-01 ~ 2025-06-30、valid 2025-07-01 ~ 2026-01-31、
   test 2026-02-01 ~ 2026-10-27，严格按时间前后切，不交叉。

## 模型

三个模型对外只有 `fit(X, y, X_valid, y_valid)` 和 `predict(X)` 两个方法，
在 `hkq/models.py` 的 `build_model` 里注册：

- **XGBoost / LightGBM**：回归未来收益，验证集早停。
- **MLP**：PyTorch，默认 `[128, 64]` + Dropout 0.2，AdamW，验证集损失早停，自动用 GPU。

加第四个模型（比如 LightGBM 的排序目标、或者 LSTM）只要实现这两个方法，再在
`build_model` 里加一行分支。

> 早停说明：标签噪声很大（5 日收益的信噪比本来就低），所以 MSE 早停在几十棵树的量级就停了。
> 换成 `label_norm: rank` 之后明显改善（XGBoost 35 棵 vs 5 棵，预测值也不再挤成几个值）。
> 想继续调就动 `models.*` 的参数和 `train.early_stopping_rounds`。

## 评估

- **Rank IC**：每个交易日算一次预测值与未来收益的 Spearman 相关，再看均值、ICIR、
  胜率、t 值。这是判断"模型有没有选股能力"最直接的指标。
- **top-k 回测**：每 5 个交易日调仓一次，等权买入预测排名前 10 只，单边成本 5 bp，
  换手按 `0.5 * Σ|w_new − w_old|` 计算；同时给出全池等权基准。
  同分的股票会先随机打散，避免默认按代码字母序选股。
- 每次训练还会输出**逐因子**的 Rank IC 表（`artifacts/alpha_ic.csv`），
  后面 psq 做因子筛选、以及看哪类因子在这个池子里真正有效时可以直接用。

**交易成本**：回测把手续费和滑点合成一个单边 `backtest.cost_bps`（默认 5 bp），
按每期换手计提：`net = gross − turnover × bps / 1e4`。换手按 `0.5 × Σ|w_new − w_old|`
算（现金也算一个仓位，所以建仓那期是 100%）。

- 美股 ETF 佣金基本为 0，这 5 bp 实际上买的是**价差 + 冲击成本**。
- 当前换手约 0.9/期（每 5 个交易日几乎换掉整个组合），年化拖累约 2.3%。
- 报告里附**成本敏感度表**（0 / 5 / 10 / 20 / 30 bp），可以直接看出策略能承受多高的摩擦。
- 注意两个口径问题：所有 ETF 统一用同一个费率，而池子里 FXF / FXB / CMBS 这类
  日均成交额只有 200–400 万美元的标的，5 bp 偏乐观；基准（全池等权）是**不计成本**的。

产物：

```
artifacts/
  alpha_ic.csv              # 每个因子的 IC / ICIR / 胜率
  report.md                 # 数据切分 + 因子 IC + 模型对比，给团队看的汇总
  <model>/predictions.parquet    # valid / test 的预测值与真实标签
  <model>/backtest_periods.csv   # 逐期收益、基准、换手
  <model>/holdings.csv           # 每期买入了哪些 ETF
  <model>/metrics.json           # IC 与绩效指标
  <model>/feature_importance.csv # 树模型的因子重要性
```

## 当前结果（示例）

跑 `main.py all` 得到（2026-10 的数据快照，仅用于验证流水线，不是策略结论）：

| model | valid IC | test IC | test CAGR | Sharpe | 最大回撤 |
| --- | --- | --- | --- | --- | --- |
| xgboost | 0.035 | 0.015 | 27.5% | 1.33 | -8.7% |
| lightgbm | 0.027 | 0.004 | 32.6% | 1.51 | -9.4% |
| mlp | 0.023 | 0.014 | 35.4% | 1.53 | -7.6% |
| 全池等权基准 | — | — | 5.5% | 0.56 | -4.7% |

IC 只有千分之几到百分之一，是这类日频横截面策略的正常量级；测试集只有 179 个交易日、
36 次调仓，收益数字的运气成分很大，别当成结论。

## 已知限制 / 下一步

1. **历史太短**：合并行情只有 2024-07 之后。要更长历史得混入 `EQUS.MINI`，
   或者只对价格类因子用单场所数据——需要团队先定口径。
2. **只有单次切分**：还没做滚动重训（walk-forward）。现在训练一次用到 2025-06，
   实际交易时应该按周或按月往后滚，这部分接口已经留好了（`split.ranges()`）。
3. **因子是临时选的**：等 psq 的因子库和筛选标准出来，替换 `dataset.alphas` 即可。
4. **持仓重叠**：ETF 池去重后仍会有重叠暴露，可以在 `dataset.py` 后面加一步
   按 `overlap_family` 做组内约束，或者在组合层面限制单方向权重。
5. **风控没接**：回测里只有交易成本，没有止损、没有波动率目标、没有仓位上限，
   这部分要等 zy 的方案。
6. **执行假设**：回测假设按收盘价全额成交，既没有按 ADV 算冲击成本，也没有部分成交和
   成交概率。日均成交额中位数在 1 亿美元以上，几百万的仓位影响不大，但仓位一放大
   就得换成带冲击模型的执行假设。