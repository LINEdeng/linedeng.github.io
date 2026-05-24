---
title: 大模型训练之—大模型预训练
date: 2026-05-02 12:00:00 +0800
categories: [llm]
tags: [llm]
toc: true
layout: post
math: true
---
> 万亿参数的智能，始于一条脏数据。

预训练是大模型的"第一性原理"——所有惊艳的对话能力、推理能力、创作能力，都根植于这个阶段注入的知识底座。一个模型的上限，在预训练结束的那一刻就已经决定了；SFT和RLHF只是 unlock，不能凭空创造。

这篇博客旨在介绍一个完整的大模型预训练的认知框架。我们从**数据获取与清洗 → Token化 → 模型训练 → 推理评估**四个阶段出发，逐层拆解预训练的完整Pipeline，覆盖语料治理、分布式训练、稳定性控制、Scaling Laws等核心议题，并对比了DeepSeek-V3、DeepSeek-R1、LLaMA 3、Qwen3四大主流模型的预训练实践——它们分别代表了MoE架构创新、RL推理范式、Over-training数据工程、超大规模数据+双模式统一四条技术路线。

## 一、总览：预训练 Pipeline 四阶段

```
Raw Internet ──▶ [Stage 1: 语料工程] ──▶ Clean Text
                        │
                        ▼
               [Stage 2: Tokenization] ──▶ Token Sequence
                        │
                        ▼
               [Stage 3: 预训练] ──▶ Base Model θ*
                        │
                        ▼
               [Stage 4: 推理] ──▶ Generated Text
```

预训练产出的 Base Model 本质是 **next-token probability estimator**，需经 Post-training（SFT/RLHF）后才可部署为对话式助手。  

---

## 二、Stage 1：语料工程

### 2.1 数据来源

| 数据源 | 规模 | 质量 | 代表数据集 |
|--------|------|------|-----------|
| 网页爬取 | PB级 | 低（需过滤） | Common Crawl, C4 |
| 书籍 | TB级 | 高 | Books3, Project Gutenberg |
| 学术论文 | TB级 | 高 | arXiv, S2ORC |
| 代码 | TB级 | 中高 | GitHub, StarCoder Data |
| 百科 | GB级 | 很高 | Wikipedia |
| 对话/论坛 | TB级 | 中 | Reddit, StackExchange |

### 2.2 专用数据

- **多语文本**：增强跨语言语义关联，提升多语理解与生成能力
- **科学文本**：arXiv论文、科学教材、数学网页，增强推理能力（需特殊分词处理公式/蛋白质序列）
- **代码**：提升结构化语义理解与逻辑推理能力，函数调用关系增强工具使用能力

### 2.3 多级数据治理 Pipeline

以 HuggingFace FineWeb 为参考实现，工业级语料治理包含以下级联阶段：

```
原始数据(PB级)
    ↓ Stage 1: URL-Level Filtering
语言过滤
    ↓ Stage 2: Content Extraction & Boilerplate Removal
正文提取
    ↓ Stage 3: Language Identification & Filtering
多语言分桶
    ↓ Stage 4: Document-Level & Passage-Level Deduplication
去重文档
    ↓ Stage 5: PII Redaction
安全数据
    ↓ 数据配比+采样
最终训练集(TB级)
```

#### Stage 1: URL-Level Filtering
- 基于 domain blocklist 粗粒度过滤
- 排除：恶意内容、低质量来源（content farms/SEO spam）、违规内容、商业噪声
- 策略倾向：**高召回率**，后续阶段可进一步过滤

#### Stage 2: Content Extraction & Boilerplate Removal
- 原始 HTML 去除导航栏/侧边栏/页脚/CSS-JS/广告脚本
- 工具：Trafilatura、Readability（基于 DOM 分析）
- 只保留正文文本

#### Stage 3: Language Identification & Filtering
- 使用 fastText 分类器识别语种
- FineWeb 仅保留英文占比 >65% 的页面
- 其他语言模型需自定义语料

#### Stage 4: 去重（Deduplication）

**Exact去重 vs Fuzzy去重：**

| 方法 | 原理 | 优势 | 劣势 |
|------|------|------|------|
| Exact（精确去重） | URL去重或SHA256哈希 | 速度极快 | 只能找完全相同 |
| Fuzzy（模糊去重） | MinHash + LSH | 找到近似重复 | 计算成本高 |
| 混合策略 | 先Exact再Fuzzy | 兼顾效率和效果 | **推荐方案** |

**MinHash 去重流程：**

1. 将文档分成 n-gram 集合
2. 用多个哈希函数计算 MinHash 签名（降维表示）
3. 用 LSH（局部敏感哈希）找到签名相似的文档对
4. Jaccard 相似度超过阈值（如0.8）视为近似重复

**为什么去重重要？** Lee et al. (2022) 发现：训练数据中重复内容越多，模型越容易逐字记忆训练数据（memorization），泛化能力下降。去重后同样 token 数训练效果更好。

#### Stage 5: PII/毒性过滤
- **PII 移除**：删除地址、电话号码、社会安全号等个人身份信息
- **毒性过滤**：移除有害、色情、暴力内容
- 符合伦理 AI 标准，防止推理时生成个人数据

#### 质量过滤具体规则

| 过滤类型 | 规则 | 目的 |
|----------|------|------|
| 长度过滤 | 文档<50词或>100K词丢弃 | 过短无信息，过长可能爬虫错误 |
| 困惑度过滤 | KenLM PPL>阈值丢弃 | 语言流畅度低的文档 |
| n-gram去重 | 文档内高比例重复n-gram丢弃 | SEO垃圾、模板页面 |
| 特殊字符 | 非字母数字比例>30%丢弃 | HTML残留、乱码 |
| 关键词黑名单 | 包含色情/赌博关键词 | 有害内容 |
| 分类器过滤 | 用Wikipedia训练的质量分类器 | GPT-3的核心方法 |

### 2.4 原始数据 vs 清洗后数据：具体示例

以下展示一条典型网页数据从原始爬取到清洗完成的完整变化过程：

**原始数据（Raw HTML + 噪声）**：
```html
<div class="article-body">
  <script>var ad_id = "XJ-29381"; track_click(ad_id);</script>
  <style>.sidebar{display:none}</style>
  <nav>首页 | 科技 | AI</nav>
  <p>机器学习是人工智能的一个分支，致力于使计算机能够从数据中<span class="highlight" style="color:red">自动学习和改进</span>，而无需进行明确的编程。</p>
  <div class="ad-banner">点击领取优惠券！限时特价！</div>
  <p>它的核心思想是通过算法分析大量数据，从中发现模式和规律，进而做出预测或决策。常见的应用包括图像识别、自然语言处理和推荐系统。</p>
  <footer>© 2024 TechNews | 隐私政策 | 广告合作 | 联系我们</footer>
</div>
<!-- SEO spam: 机器学习 机器学习 机器学习 机器学习 机器学习 -->
```

**清洗后文本**：
```
机器学习是人工智能的一个分支，致力于使计算机能够从数据中自动学习和改进，而无需进行明确的编程。它的核心思想是通过算法分析大量数据，从中发现模式和规律，进而做出预测或决策。常见的应用包括图像识别、自然语言处理和推荐系统。
```

**具体清洗操作对照**：

| 操作 | 处理内容 | 对应Pipeline阶段 |
|------|----------|-----------------|
| HTML标签去除 | 移除`<div>`, `<script>`, `<style>`, `<nav>`, `<footer>`等 | Stage 2: 正文提取 |
| 广告/导航剥离 | 去掉"点击领取优惠券"、导航栏、页脚 | Stage 2: Boilerplate Removal |
| 样式信息清理 | 移除`style="color:red"`, `class="highlight"` | Stage 2: 正文提取 |
| SEO垃圾去除 | 删除末尾关键词堆叠 | Stage 4: n-gram去重 |
| 格式还原 | `<span>`内文本还原为纯文本 | Stage 2: 正文提取 |
| 标记清理 | 去掉HTML实体、注释`<!-- -->` | Stage 2: 正文提取 |

### 2.5 数据配比

不同来源的混合比例对模型能力有显著影响：

| 配比策略 | 方法 | 效果 |
|----------|------|------|
| 均匀采样 | 各来源等比例 | 基线 |
| 启发式配比 | 人工设定比例 | LLaMA-1 用了这个 |
| DoReMi | 用小模型自动学习最优配比 | 比人工好 |
| Scaling Laws辅助 | 小规模实验确定配比 | 最科学 |

**LLaMA 的数据配比参考：**

| 来源 | 比例 |
|------|------|
| CommonCrawl | 67% |
| C4 | 15% |
| GitHub | 4.5% |
| Wikipedia | 4.5% |
| Books | 4.5% |
| arXiv | 2.5% |
| StackExchange | 2% |

### 2.6 数据课程（Data Curriculum）

通过在多个阶段使用不同数据配比来增强特定能力：

| 目标能力 | 代表工作 | 数据课程 |
|----------|----------|----------|
| 代码 | CodeLLaMA | 2T通用 → 500B代码密集 |
| Python代码 | CodeLLaMA-Python | 2T通用 → 500B代码 → 100B Python |
| 数学 | Llemma | 2T通用 → 500B代码 → 50~200B数学（含5%通用数据正则化） |
| 长文本 | CodeLLaMA | 4K上下文2.5T → 16K上下文20B |

---

## 三、Stage 2：Tokenization

### 3.1 核心Trade-off

Sequence length 是 Transformer 最稀缺的计算资源（self-attention 时空复杂度 O(n²)），需在 vocabulary size 与 sequence length 间找 Pareto-optimal 平衡点。序列 token 数量小幅增长就会带来算力、显存开销的平方级飙升，极大限制模型处理能力。词汇表规模和序列长度彼此相互制约，扩充词表能让单个 token 承载更多语义信息，缩减序列长度、降低计算压力，但会造成嵌入层参数增多、低频词学习难度上升、解码成本增加；压缩词表虽能精简模型词库，却会拆分出更多 token 拉长序列，大幅拉高运算损耗，因此需要在两者之间寻找帕累托最优平衡点，兼顾模型效果、运行效率与硬件成本，实现整体综合收益最大化。

```
Vocabulary Size ↑  ⟺  Sequence Length ↓  ⟺  计算效率 ↑
Vocabulary Size ↓  ⟺  Sequence Length ↑  ⟺  计算效率 ↓
```

### 3.2 编码层级与压缩递进

| Level | 表示 | Vocabulary Size | 压缩比 |
|-------|------|----------------|--------|
| L0 | Raw Unicode Text | ~150K codepoints | 1x |
| L2 | Byte Sequence | 256 | 1x |
| L3 | BPE Tokens | ~100K | ~4x |

### 3.3 Byte Pair Encoding（BPE）算法

当前工业标准。核心思想：迭代合并高频相邻符号对构建 vocabulary。

**算法流程：**
```
Input: byte sequence B, initial vocab V = {0, 1, ..., 255}
Repeat K times (~100,000):
  1. 统计 B 中所有 adjacent pair 的频次
  2. 选取频次最高的 pair (bₐ, bᵦ)
  3. 创建新符号 s = |V|，加入 V
  4. 将 B 中所有 (bₐ, bᵦ) 替换为 s
Output: compressed sequence B', vocabulary V with |V| = 256 + K
```

- GPT-4 tokenizer (cl100k_base) vocabulary size = 100,277
- 常见词为单一 token，罕见词被拆分为多个 subword units
- 前导空格通常编码进 token 内部
- Case-sensitive：`"Hello"` 与 `"hello"` 映射到不同 token ID

### 3.4 中文 Tokenizer 注意事项

- 1个token约对应1.5个汉字比较合理
- 建议数字拆分逻辑（避免"9.9 vs 9.11"问题）
- 词表预留约1000位置供对齐阶段引入新token
- 清理脏数据（GPT-4o词表泄露事件教训）
- 垂直领域可提前设高频词为独立token

### 3.5 Token化结果：具体数据示例

以"机器学习是人工智能的一个分支"为例，展示从原始文本到模型输入张量的完整变换过程：

**Step 1：文本分块（Chunking）**
```
原始长文本 → 按max_seq_length切分为固定长度片段
特殊token添加：[BOS] 机器学习是人工智能的一个分支 [EOS] [PAD] [PAD] ...
```

**Step 2：Tokenizer编码**
```
原始文本:    "机器学习是人工智能的一个分支"
Token序列:   ["机器", "学习", "是", "人工", "智能", "的", "一个", "分支"]
Token IDs:   [5821, 1492, 322, 8434, 3120, 319, 428, 7103]
```

**Step 3：构造模型输入张量**

对于因果语言建模（CLM），labels即为input_ids的右移版本：

```python
# 输入: "机器学习是人工智能的一个分支"

# input_ids: 模型看到的token序列
input_ids      = [1,    5821, 1492, 322,  8434, 3120, 319,  428,  7103, 2]
#               [BOS]  机器   学习   是    人工   智能   的    一个   分支   [EOS]

# labels: 模型要预测的目标（input_ids左移一位）
labels         = [5821, 1492, 322,  8434, 3120, 319,  428,  7103, 2,    -100]
#               机器   学习   是    人工   智能   的    一个   分支   [EOS] [ignore]

# attention_mask: 标记有效token位置
attention_mask = [1,    1,    1,    1,    1,    1,    1,    1,    1,    0]
#               有效   有效   有效  有效   有效   有效   有效   有效   有效   填充
```

**关键设计说明**：

| 字段 | 作用 | 关键细节 |
|------|------|----------|
| `input_ids` | 模型输入的token索引序列 | 起始加[BOS]=1，结尾加[EOS]=2 |
| `labels` | 计算损失的目标序列 | 等于input_ids右移一位；-100（ignore_index）处不计算损失 |
| `attention_mask` | 标记哪些位置是真实token | 0=padding位置，不参与attention计算 |
| `[BOS]` | 句首标记 | Token ID通常为1，提示模型"一段新文本开始" |
| `[EOS]` | 句尾标记 | Token ID通常为2，模型学会在此时停止生成 |
| `[PAD]` | 填充标记 | 短文本填充至固定长度，使同一batch形状一致 |
| `-100` | 忽略标记 | PyTorch CrossEntropyLoss的ignore_index，PAD和EOS之后位置设为-100 |

**CLM训练的本质**：每个位置的目标就是预测下一个token：
```
位置1: 看到[BOS]        → 预测"机器"(5821)
位置2: 看到[BOS]机器     → 预测"学习"(1492)
位置3: 看到[BOS]机器学习  → 预测"是"(322)
...
位置8: 看到[BOS]...一个   → 预测"分支"(7103)
位置9: 看到[BOS]...分支   → 预测[EOS](2) ← 学会停止
```

参考：[大规模语言模型预训练全链路深度解析](https://blog.csdn.net/luosuss/article/details/159616668)、[Wikipedia - Large language model](https://en.wikipedia.org/wiki/Large_language_model)

---

## 四、Stage 3：预训练

### 4.1 训练目标

#### 因果语言建模（CLM）— 主流方案
- 代表模型：GPT系列、LLaMA、PaLM
- 逻辑：给定前序文本，预测下一个Token（自回归生成）
- 目标函数：
  $$
  L = -\sum_{t=1}^{T} \log P(x_t \mid x_{<t})
  $$
  $L = -\sum_{t=1}^{T} \log P(x_t | x_{<t})$
- 单向上下文（通过因果掩码实现），擅长文本生成

#### 掩码语言建模（MLM）— BERT系
- 代表模型：BERT
- 逻辑：随机掩盖15%的Token，让模型预测被掩盖的Token
- 双向理解上下文，适合理解任务（分类、NER等）

### 4.2 模型架构

| 特性 | Decoder-Only（GPT类） | Encoder-Decoder（T5类） |
|------|----------------------|------------------------|
| 架构 | 仅解码器堆栈 | 完整编码器+解码器 |
| 注意力掩码 | 因果掩码（单向） | 编码器：无掩码（双向）；解码器：因果掩码 |
| 输入-输出 | 单序列生成 | 双序列映射 |
| 典型应用 | 开放生成、对话、续写 | 任务型生成（翻译、摘要） |

主流预训练模型基本采用 **Decoder-Only** 架构。

### 4.3 分布式训练策略

这是预训练中最复杂的工程环节，千亿参数模型需要千卡级集群，以下简单介绍，后续出大模型分布式训练的帖子来详谈。

#### 4.3.1 数据并行（Data Parallelism）

| 方案 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| DP | 多GPU复制模型，GPU 0聚合梯度 | 实现简单 | GPU 0瓶颈，显存冗余 |
| DDP | 每个GPU直接处理mini-batch，梯度AllReduce同步 | 支持多节点，更高效 | 每GPU仍存完整模型 |

#### 4.3.2 ZeRO 系列（Zero Redundancy Optimizer）

| 阶段 | 分片内容 | 内存节省 | 通信量 |
|------|----------|----------|--------|
| ZeRO-1 | 优化器状态 | ~4x | 2Ψ |
| ZeRO-2 | 优化器状态+梯度 | ~8x | 2Ψ |
| ZeRO-3 | 参数+梯度+优化器状态全分片 | N倍(N=GPU数) | 3Ψ |

- **ZeRO-Offload**：将优化器和部分计算offload到CPU
- **ZeRO-Infinity**：进一步扩展到NVMe存储，支持万亿参数
- **ZeRO++**：量化权重通信优化

#### 4.3.3 张量并行（Tensor Parallelism）
- 将单个算子（如Linear层）按维度切分到多个GPU
- Megatron-LM的Column-wise / Row-wise切分
- 适合单节点内多GPU（需高带宽NVLink互连）

#### 4.3.4 流水线并行（Pipeline Parallelism）
- 将模型按层切分到不同GPU（如前12层GPU0，后12层GPU1）
- 需要微批次流水线（GPipe / 1F1B调度）减少气泡
- 适合跨节点场景

#### 4.3.5 混合并行（Hybrid Parallelism）— 工业界主流

典型方案：**3D并行 = 数据并行 + 张量并行 + 流水线并行**

| 策略 | 适用模型规模 | 千卡吞吐提升(vs单卡) | 通信开销占比 |
|------|-------------|---------------------|-------------|
| 数据并行 | <1B参数 | ~850x | ~32% |
| 张量+流水线并行 | 1B-100B参数 | ~920x | ~18% |
| MoE+动态专家路由 | >100B参数 | ~960x | <12% |

**框架选择建议：**
- <10亿参数 + 大数据 → PyTorch DDP
- 10亿-100亿参数单卡装不下 → FSDP / ZeRO
- >100亿参数 + 10TB数据 → DeepSpeed / Megatron-LM 3D并行

### 4.4 训练核心技巧

#### 4.4.1 混合精度训练
- **FP16**：前向/反向用FP16计算，参数更新用FP32主拷贝 + Loss Scaling防止下溢
- **BF16**：A100+显卡支持，无需Loss Scaling（指数范围与FP32相同）
- 效果：显存减少约50%，计算速度提升2-3倍

#### 4.4.2 梯度累积
- 显存不足时，用小batch多次前向/反向，累积梯度后再更新
- 等效batch_size = micro_batch_size × accumulation_steps
- 数学上等价于一次大batch训练

#### 4.4.3 梯度检查点（Activation Checkpointing）
- 前向时只保存部分激活值，反向时重计算丢弃的激活值
- 以计算换显存，典型节省50%+显存

#### 4.4.4 FlashAttention-2
- 长序列（>4K）必开，显存省50%+，速度升2倍以上
- 通过IO感知的注意力计算，减少HBM读写

### 4.5 学习率调度

#### 主流方案对比

| 调度策略 | 优点 | 缺点 | 应用场景 |
|----------|------|------|----------|
| 恒定学习率 | 简单直观 | 难以平衡初期收敛与后期优化 | 无 |
| 余弦衰减(Cosine Decay) | 优化稳定，泛化性好 | 需预设总训练步数 | 预训练标准方案 |
| Warmup+Cosine Decay | 兼顾稳定性与优化效率 | 参数较多 | 所有训练场景 |
| WSD(Warmup-Stable-Decay) | 灵活，可在任意时刻结束训练 | 需调衰减起点 | 大规模预训练新趋势 |

#### 关键参数配置参考（以TinyLlama为例）

```yaml
optimizer: AdamW
lr: 4e-4           # 初始学习率
min_lr: 4e-5       # 最小学习率（初始的10%）
warmup_steps: 2000  # 预热步数
weight_decay: 0.1
betas: [0.9, 0.95]
grad_clip: 1.0      # 梯度裁剪
```

#### Scaling Laws 与学习率

最新研究表明，最优学习率与模型大小N、数据量D满足幂律关系：
$$
Lr(N, D) = 38.4588 \cdot N^{-0.2219} \cdot D^{-0.3509}
$$
参考：[How to Set the Learning Rate for Large-Scale Pre-training?](https://openreview.net/pdf?id=SrJEEWSg1c)

### 4.6 训练稳定性：Loss Spike 问题

#### 什么是Loss Spike？
大模型（100B+）预训练过程中loss突然暴涨，恢复缓慢甚至永远无法收敛。

#### 根本原因
- Adam优化器中梯度变化与参数更新不满足独立性
- 浅层参数长时间不更新，后期梯度又趋于平稳
- Batch太大加剧问题
- 梯度异常值（gradient spikes）通过动量机制的指数平均产生持续影响

#### 解决方案

| 方法 | 原理 | 效果 |
|------|------|------|
| 回退Checkpoint | 找到spike前的checkpoint重新训练 | PaLM/GLM-130B使用 |
| 梯度裁剪 | 限制梯度范数上限 | 基础手段 |
| Embed LN | 对Embedding层加LayerNorm | 控制梯度范数上界 |
| Scaled Embed | Embedding乘√d | 满足梯度上界控制条件 |
| Adam ε调大 | 增大优化器分母防止除零 | 直接有效 |
| SPAM | Spike-Aware Adam + 动量重置 | 新方法，实验效果好 |

参考：[Spike No More: Stabilizing the Pre-training of Large Language Models](https://arxiv.org/html/2312.16903)、[SPAM: Spike-Aware Adam](https://openreview.net/pdf?id=L9eBxTCpQG)

### 4.7 Checkpoint管理

- 定期保存（如每1000步），包含：模型参数、优化器状态、训练步数、LR scheduler状态
- 支持断点续训（resume from checkpoint）
- 多副本保存防止存储故障
- 异步保存减少训练中断

### 4.8 训练监控实例

以下展示一个7B模型预训练过程中典型的监控数据：

**训练日志示例（NanoQwen-7B, 8层Transformer Block, batch_size=32）**：

| Epoch | Training Loss | Validation Loss | Learning Rate | Tokens/s/GPU | MFU | Time |
|-------|--------------|----------------|---------------|-------------|-----|------|
| 1 | 3.847 | 3.521 | 3.0e-4 (warmup) | 4,200 | 42% | 2h 15min |
| 2 | 2.916 | 2.803 | 2.8e-4 | 5,100 | 51% | 2h 08min |
| 3 | 2.438 | 2.401 | 2.4e-4 | 5,400 | 54% | 2h 05min |
| 5 | 1.982 | 2.014 | 1.6e-4 | 5,600 | 56% | 2h 03min |
| 8 | 1.693 | 1.782 | 8.0e-5 | 5,700 | 57% | 2h 01min |
| 10 | 1.547 | 1.721 | 4.0e-5 (min_lr) | 5,700 | 57% | 2h 01min |

**关键监控指标解读**：

| 指标 | 含义 | 正常范围 | 异常信号 |
|------|------|----------|----------|
| Training Loss | 训练集上的交叉熵损失 | 持续下降，后期收敛 | 突然暴涨→Loss Spike；不下降→学习率/数据问题 |
| Validation Loss | 验证集上的损失 | 先降后趋平 | 持续升高→过拟合；与Training Loss差距大→泛化差 |
| Learning Rate | 当前学习率 | 按Cosine/WSD调度变化 | 降为0太早→欠拟合；一直很大→不收敛 |
| Tokens/s/GPU | 单GPU吞吐量 | 越高越好 | 显著低于预期→数据加载瓶颈/通信瓶颈 |
| MFU | 模型算力利用率 | 40%-60%正常，>50%优秀 | <30%→并行效率低/IO瓶颈 |
| 梯度范数(Grad Norm) | 参数梯度L2范数 | 稳定在1-10范围 | 突然爆炸→Loss Spike前兆；趋近0→梯度消失 |

**Loss Spike应对决策树**：
```
Loss突然暴涨
    ├─ 梯度范数 > 10× 正常值？
    │    ├─ 是 → 梯度裁剪 + 降低学习率 → 观察100步
    │    └─ 否 → 检查数据批次（是否有异常样本）
    │
    ├─ 100步后loss是否恢复？
    │    ├─ 是 → 记录事件，继续训练
    │    └─ 否 → 回退到最近正常checkpoint重新训练
    │
    └─ 频繁出现spike？
         ├─ 是 → 启用Embed LN / 调大Adam ε / 检查数据质量
         └─ 否 → 正常波动，无需处理
```

---

## 五、Scaling Laws：算力最优分配

### 5.1 Kaplan Scaling Laws（OpenAI, 2020）

$$ L(N) \propto N^{-0.076} $$

模型越大性能越好，但收益递减。
**问题**：让行业过度追求"堆参数"，做出了175B但只训300B tokens的GPT-3。

### 5.2 Chinchilla Scaling Laws（DeepMind, 2022）— 重大修正

$$ L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}} $$

其中 α ≈ 0.34, β ≈ 0.28

**核心结论：模型规模和数据规模应同步增长**

$$ N_{opt} \propto C^{0.5}, \quad D_{opt} \propto C^{0.5} $$

- 每个参数配约20个tokens（D ≈ 20N）才是计算最优
- Chinchilla 70B + 1.4T tokens 打败 Gopher 280B + 300B tokens

### 5.3 过训练趋势（Over-training）

推理成本按参数量计，而训练是一次性的 → **小模型+更多数据** 更经济

| 模型 | 参数量 | 训练Token | Token/参数比 |
|------|--------|-----------|-------------|
| GPT-3 | 175B | 300B | ~1.7 |
| Chinchilla | 70B | 1.4T | 20 |
| LLaMA-2 | 70B | 2T | ~28 |
| LLaMA-3 | 70B | 15T | ~214 |

参考：[Chinchilla scaling laws](https://aiwiki.ai/wiki/chinchilla_scaling)、[Kaplan定律 vs Chinchilla定律](https://blog.csdn.net/Qnesp/article/details/158540585)

---

## 六、工业级预训练配置示例

### 典型7B模型预训练配置

```yaml
# 模型架构
model:
  type: Decoder-Only Transformer
  hidden_size: 4096
  num_layers: 32
  num_attention_heads: 32
  max_sequence_length: 4096

# 训练配置
training:
  total_tokens: 1T~2T
  batch_size: 1024 (global)
  micro_batch_size: 2
  gradient_accumulation_steps: 8
  
# 优化器
optimizer: AdamW
  lr: 3e-4
  min_lr: 3e-5
  weight_decay: 0.1
  betas: [0.9, 0.95]
  grad_clip: 1.0
  
# 学习率调度
scheduler: Cosine with Linear Warmup
  warmup_steps: 2000
  
# 精度与并行
mixed_precision: bf16
parallel: FSDP + FlashAttention-2
  
# Checkpoint
save_interval: 1000 steps
resume: true

# 收敛目标
target_loss: 1.2~1.5 (预训练)
```

参考：[大模型分布式训练实战手册](https://blog.csdn.net/InitFlow/article/details/160054257)、[从零构建大模型](https://blog.csdn.net/ting9452000/article/details/160382831)

---

## 七、Stage 4：推理测试与评估

预训练完成后，必须通过系统化测试验证模型能力，再决定是否进入Post-training阶段。

### 7.1 模型加载

```python
# 1. 加载预训练权重
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "./checkpoints/epoch-10/",
    torch_dtype=torch.bfloat16,       # 推理用BF16精度
    device_map="auto",                 # 自动分配GPU
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("./checkpoints/epoch-10/")

# 2. 设置评估模式
model.eval()                           # 关闭Dropout、BatchNorm等训练行为
# 注意：推理时不需要梯度，无需model.train()

# 3. 可选：量化加速推理
# model = model.quantize(4)           # 4-bit量化，推理速度提升2-3倍
```

### 7.2 推理测试

```python
# 自回归生成
input_text = "人工智能的未来发展"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

with torch.no_grad():                  # 关闭梯度计算，节省显存
    outputs = model.generate(
        input_ids,
        max_new_tokens=200,            # 最大生成长度
        temperature=0.8,               # 控制随机性（0=贪心，1=标准采样）
        top_k=50,                      # Top-K采样
        top_p=0.95,                    # Nucleus采样
        do_sample=True,                # 启用随机采样
        repetition_penalty=1.1         # 重复惩罚
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
# 输出示例：
# "人工智能的未来发展将深刻改变人类社会的方方面面。从医疗诊断到自动驾驶，
#  从科学研究到创意创作，AI技术正在各个领域展现出前所未有的潜力。
#  然而，我们也需要关注AI伦理、数据隐私和技术安全等重要议题..."
```

**生成策略对比**：

| 策略 | 原理 | 适用场景 | 缺点 |
|------|------|----------|------|
| Greedy | 每步选概率最高的token | 事实性问答、代码 | 重复、缺乏多样性 |
| Top-K | 从概率最高的K个token中采样 | 通用生成 | K值需手动调，不同场景最优K不同 |
| Top-P (Nucleus) | 选累积概率≥P的最小token集 | 通用生成（最常用） | 自适应选择范围 |
| Temperature | 调整logits分布的尖锐度 | 控制创造性 | T→0≈贪心，T→∞≈均匀随机 |
| Beam Search | 保留top-B序列全局搜索 | 翻译、摘要 | 生成缓慢，可能过于保守 |

### 7.3 性能评估

#### 7.3.1 评估指标体系

| 评估维度 | 指标 | 测量方法 | 基线参考（7B模型） |
|----------|------|----------|-------------------|
| **推理速度** | Tokens/s | 单GPU生成吞吐量 | 30-80 tokens/s (BF16) |
| **推理速度** | 首Token延迟(TTFT) | 输入处理到第一个token输出的时间 | <200ms (4K上下文) |
| **内存占用** | 显存峰值 | 推理时GPU显存使用量 | ~14GB (BF16, 7B) |
| **内存占用** | KV Cache大小 | 长序列生成时缓存占用 | ~2GB/32K tokens |
| **语言质量** | Perplexity (PPL) | 测试集上的困惑度 | 5-8 (英文wiki) |
| **语言质量** | BLEU | 与参考文本的n-gram重合度 | 0.2-0.4 (开放生成) |
| **知识能力** | 准确率 | MMLU/C-Eval/ARC等benchmark | MMLU: 45-55% (7B base) |
| **代码能力** | Pass@1 | HumanEval代码生成通过率 | 20-35% (7B base) |
| **数学推理** | 准确率 | GSM8K/MATH | GSM8K: 30-50% (7B base) |
| **长文本** | 检索准确率 | Needle-in-Haystack | >95% (128K上下文) |

#### 7.3.2 Base Model核心评估集

| 评估集 | 语言 | 测试能力 | 样本量 |
|--------|------|----------|--------|
| MMLU | 英 | 多学科知识(57领域) | 14,042 |
| C-Eval | 中 | 中文多学科知识 | 13,948 |
| ARC-Challenge | 英 | 科学推理 | 1,172 |
| HellaSwag | 英 | 常识推理 | 10,042 |
| WinoGrande | 英 | 共指消解 | 2,634 |
| HumanEval | 代码 | Python代码生成 | 164 |
| GSM8K | 英 | 数学推理(小学) | 1,319 |
| MATH | 英 | 数学推理(竞赛) | 5,000 |

#### 7.3.3 评估流程

```
预训练完成
    ↓
Step 1: PPL评估
    在保留的验证集上计算困惑度，确认模型收敛
    目标：PPL < 预设阈值（如英文wiki PPL < 8）
    ↓
Step 2: Benchmark评估
    运行标准评估集（MMLU/C-Eval/HumanEval/GSM8K等）
    对比同规模基线模型（如LLaMA-7B, Qwen-7B）
    ↓
Step 3: 长上下文评估
    Needle-in-Haystack测试
    不同长度(4K/8K/32K/128K)下的检索准确率
    ↓
Step 4: 生成质量评估
    人工评估 + GPT-4评判（流畅性、连贯性、事实性）
    与基线模型对比（ELO评分或胜率）
    ↓
Step 5: 安全性评估
    有害内容生成率、偏见测试、PII泄露检测
    ↓
通过评估 → 进入Post-training（SFT/RLHF/DPO）
未通过   → 分析弱点，补充数据续训或调整架构
```

---

## 八、关键挑战与前沿趋势

### 8.1 当前挑战
1. **数据墙**：高质量数据即将耗尽，LLM生成内容污染训练数据
2. **训练成本**：GPT-3级约4600万美元，千卡集群月级别训练
3. **训练稳定性**：loss spike、数值溢出、硬件故障
4. **知识时效性**：预训练数据截止时间固定，无法实时更新

### 8.2 前沿趋势
1. **合成数据**：用强模型生成高质量训练数据
2. **MoE架构**：稀疏激活，同等性能下降低推理成本
3. **自动化并行**：AI自动选择最优并行策略（如Google Alpa）
4. **异构计算**：GPU+CPU+TPU混合集群协同训练
5. **WSD调度**：更灵活的学习率方案，适应动态训练规模
6. **Over-training**：小模型+超量数据，降低推理成本

---

## 九、总结：预训练全流程速查

```
1. 数据采集 → 网页/书籍/论文/代码/百科/论坛
2. URL过滤 → blocklist排除恶意/低质量/违规源
3. 正文提取 → HTML→纯文本，去导航/广告/CSS-JS
4. 语言过滤 → fastText识别，保留目标语言
5. 质量过滤 → 启发式规则 + 困惑度 + 分类器
6. 去重 → Exact(SHA256) + Fuzzy(MinHash+LSH)
7. 安全过滤 → PII移除 + 毒性过滤
8. 数据配比 → 启发式/DoReMi/Scaling Laws确定混合比例
9. Tokenization → BPE/SentencePiece训练分词器
10. 构造张量 → input_ids + labels + attention_mask
11. 模型构建 → Decoder-Only Transformer
12. 分布式训练 → 3D并行(DP+TP+PP) + ZeRO + FlashAttention
13. 优化 → AdamW + Warmup+Cosine/WSD + 梯度累积 + 混合精度
14. 稳定性 → 梯度裁剪 + Embed LN + Checkpoint回退
15. 监控 → Loss曲线、梯度范数、算力利用率(MFU>70%)
16. 推理测试 → 模型加载(eval模式) + 自回归生成 + 生成策略(Top-P/Temperature)
17. 性能评估 → PPL/BLEU/MMLU/HumanEval/GSM8K + 长上下文Needle测试 + 安全性评估
```

---

## 十、主流大模型的预训练实践

### 10.1 DeepSeek-V3（2024.12）

**定位**：671B参数 MoE 大模型，每个token仅激活37B参数（约5.5%），兼顾性能与成本效率。

**核心架构创新**：

| 组件 | 技术 | 核心作用 |
|------|------|----------|
| 注意力 | MLA（多头潜在注意力） | KV缓存低秩压缩至512维，压缩率97%，推理显存降至1/30 |
| FFN | DeepSeekMoE | 256个路由专家 + 共享专家，top-8激活 |
| 负载均衡 | 无辅助损失策略 | 动态偏置路由，替代传统auxiliary loss，避免性能下降 |
| 训练目标 | MTP（多Token预测） | 预测D+1个未来token，提升数据效率，可用于推测解码加速 |

**模型配置**：
- 61层Transformer，前3层稠密，后续每2层嵌入MoE结构
- 隐藏维度7168，128个注意力头，KV压缩维度512
- 128K词表（BPE），RoPE位置编码

**预训练数据**：14.8T高质量多样化token

**训练基础设施**：
- 硬件：2048张 NVIDIA H800 GPU（80GB），8卡/节点
- 并行策略：16路流水线并行 + 64路专家并行 + ZeRO-1数据并行
- **DualPipe算法**：计算-通信完全重叠，跨节点All-to-All通信开销降至接近零
- **FP8混合精度训练**：首次在超大规模模型上验证成功
  - 大部分GEMM用FP8（E4M3），关键操作保留BF16/FP32
  - 激活按1×128分块量化，权重按128×128分块量化
  - 每累加128个元素即提升至FP32累加，减少精度损失
  - 优化器动量用BF16存储，MoE通信中激活用FP8（节省50%带宽）

**训练稳定性**：整个训练过程无不可恢复的loss spike，无需回滚

**训练成本**：仅2.788M H800 GPU小时，远低于同规模模型

> 参考：[DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437)、[HuggingFace DeepSeek-V3](https://huggingface.co/docs/transformers/main/model_doc/deepseek_v3)

---

### 10.2 DeepSeek-R1（2025.01）

**定位**：基于DeepSeek-V3-Base的推理增强模型，首次证明纯RL可让LLM自主发展推理能力。

**核心创新**：GRPO（组相对策略优化）+ 多阶段训练管道

**训练流程**：

```
DeepSeek-V3-Base
       │
       ├──── R1-Zero路径（纯RL）──────────────────────┐
       │    无SFT，直接GRPO强化学习                    │
       │    奖励：准确性（规则验证）+ 格式               │
       │    结果：AIME 2024 从15.6% → 71.0%           │
       │    问题：可读性差、语言混杂                     │
       │                                             │
       └──── R1路径（冷启动+多阶段）────────────────────┘
            │
            ▼
       Stage 1: 冷启动SFT
            数千条长CoT数据微调V3-Base
            格式：<think推理过程</think摘要
            │
            ▼
       Stage 2: 推理导向RL
            GRPO + 准确性奖励 + 格式奖励 + 语言一致性奖励
            聚焦数学/代码/科学/逻辑推理
            │
            ▼
       Stage 3: 拒绝采样 + SFT
            ~60万推理样本（从RL checkpoint采样+过滤）
            + ~20万非推理样本（写作/问答/翻译等）
            合计约80万样本，2轮微调
            │
            ▼
       Stage 4: 全场景RL
            推理数据：规则化奖励
            通用数据：模型化奖励（helpfulness + harmlessness）
            │
            ▼
       DeepSeek-R1
```

**GRPO vs PPO**：
- GRPO无需训练Critic网络，通过组内采样（每组G个输出）标准化奖励计算优势函数

- 优势函数：
  $$
  A_i = \frac{r_i - \mu}{\sigma}
  $$
  （组内均值-方差标准化）

- 显著减少内存和计算成本

**"顿悟时刻"（Aha Moment）**：

- 训练约8000步后，模型自发学会反思——重新评估初始方法
- 反思性token（如"wait"、"check"、"verify"）频率增长5-7倍
- 推理链长度持续增长至数千token，体现测试时计算扩展

**训练成本**：约147K H800 GPU小时（~$294K），为OpenAI O1的1/80

**蒸馏**：用R1生成的80万条推理数据蒸馏Qwen/LLaMA小模型，R1-Distill-Qwen-32B在AIME上达72.6%

> 参考：[DeepSeek-R1 Nature Paper](https://www.nature.com/articles/s41586-025-09422-z)、[DeepSeek-R1论文全文翻译](https://blog.csdn.net/wshzd/article/details/145515412)

---

### 10.3 LLaMA 3 / 3.1（2024.04 / 2024.07）

**定位**：Meta开源Dense Transformer模型家族，旗舰405B参数，代表Over-training范式。

**模型规格**：

| 模型 | 参数量 | 层数 | 隐藏维度 | 注意力头 | KV头 | 上下文 | 训练Token |
|------|--------|------|----------|----------|------|--------|-----------|
| LLaMA 3 8B | 8B | 32 | 4,096 | 32 | 8 | 8K→128K | 15T+ |
| LLaMA 3 70B | 70B | 80 | 8,192 | 64 | 8 | 8K→128K | 15T+ |
| LLaMA 3.1 405B | 405B | 126 | 16,384 | 128 | 8 | 8K→128K | 15.6T |

**架构特点**：
- **Dense Transformer**（非MoE），坚持简单架构管理复杂度
- **GQA（Grouped-Query Attention）**：所有模型KV头统一为8，提升推理效率
- **128K词表**（BPE），比LLaMA 2的32K大4倍
- RoPE基础频率提升至500,000
- 注意力掩码防止不同文档间的self-attention

**预训练数据处理**：

```
原始网页数据
    ↓ PII/安全过滤
    ↓ 文本提取与清洗（保留数学公式、代码块结构）
    ↓ 三级去重：URL级 → 文档级(MinHash) → 行级(>6次出现)
    ↓ 启发式过滤：n-gram覆盖率、"脏词"计数、KL散度异常检测
    ↓ 模型质量过滤：fasttext + DistilRoberta（基于LLaMA 2标注训练）
    ↓ 代码/推理专用Pipeline
    ↓ 多语言Pipeline（176种语言，fasttext识别）
最终训练集
```

**数据配比**（通过知识分类 + Scaling Law实验确定）：

| 数据类别 | 比例 |
|----------|------|
| 通用知识 | 50% |
| 数学与推理 | 25% |
| 代码 | 17% |
| 多语言 | 8% |

**三阶段训练方案**：

1. **Initial Pre-training**
   - AdamW优化器，peak LR: 8×10⁻⁵（405B），warmup 8000步
   - Cosine LR schedule衰减至peak LR的1/30
   - 动态batch size：4M tokens(4K) → 8M(8K) → 16M(8K)
   - 训练过程中调整数据配比（增加非英语/数学/最新网络数据）

2. **Long-Context Pre-training**
   - 6个阶段逐步扩展上下文：8K → 128K
   - 总计约0.8T tokens的长上下文训练
   - 评估标准：短上下文性能不退化 + Needle-in-Haystack任务通过

3. **Annealing**
   - 最后40M tokens线性退火学习率至0
   - 上采样高质量代码和数学数据
   - 8B模型在GSM8K提升24%、MATH提升6.4%（405B提升可忽略）

**Over-training关键发现**：LLaMA 3 8B的Chinchilla最优数据量为200B tokens，但训练至15T（75倍）仍持续log-linear提升

**训练成本**：7.7M H100-80GB GPU小时，碳排2290 tCO2eq（100%碳抵消）

> 参考：[The Llama 3 Herd of Models](https://arxiv.org/pdf/2407.21783)、[Llama 3 Model Card](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md)

---

### 10.4 Qwen3（千问3）（2025.04）

**定位**：阿里通义千问系列最新版，36T token超大规模预训练，思考/非思考双模式统一。

**模型家族**：

| 模型 | 类型 | 层数 | 注意力头(Q/KV) | 专家(Total/Active) | 上下文 |
|------|------|------|----------------|-------------------|--------|
| Qwen3-0.6B | Dense | 28 | 16/8 | - | 32K |
| Qwen3-1.7B | Dense | 28 | 16/8 | - | 32K |
| Qwen3-4B | Dense | 36 | 32/8 | - | 32K |
| Qwen3-8B | Dense | 36 | 32/8 | - | 128K |
| Qwen3-14B | Dense | 40 | 40/8 | - | 128K |
| Qwen3-32B | Dense | 64 | 64/8 | - | 128K |
| Qwen3-30B-A3B | MoE | 48 | 32/4 | 128/8 | 128K |
| Qwen3-235B-A22B | MoE | 94 | 64/4 | 128/8 | 128K |

**架构创新**：
- **QK-Norm**：注意力机制中引入QK归一化，提升训练稳定性
- 移除QKV偏置（Qwen2中有，Qwen3中去掉）
- MoE：**128个路由专家，8个激活，无共享专家** + 全批次负载均衡损失
- 思考/非思考模式统一：一个模型兼顾推理与快速响应
- Thinking Budget机制：根据问题复杂度动态分配推理计算量

**预训练数据**：约36T token，覆盖119种语言（Qwen2.5仅29种），为当前开源模型最大规模

**数据扩展策略**：
- Qwen2.5-VL从PDF文档提取文本
- Qwen2.5-Math生成数学合成数据（教科书、问答对）
- Qwen2.5-Coder生成代码合成数据（代码片段）
- 两阶段过滤：Qwen2.5-72B-Instruct过滤低质量数据 → 人工评估候选响应
- 实例级数据混合优化（非传统领域级混合），消融实验在小模型上验证

**三阶段预训练**：

```
Stage 1 (S1): 通用基础
    数据：30T+ tokens，119种语言
    序列长度：4096
    目标：基础语言能力 + 通用世界知识

Stage 2 (S2): 推理增强
    数据：5T tokens，知识密集型（STEM/编码/推理）
    序列长度：4096
    技术要点：加速学习率衰减，增加STEM和编码数据比例

Stage 3: 长上下文扩展
    数据：数百亿tokens长文本
    序列长度：扩展至32768
    数据构成：75%长度16K-32K，25%长度4K-16K
    技术要点：
      - ABF技术：RoPE基础频率 10,000 → 1,000,000
      - YaRN + DCA（双块注意力）：推理时序列容量提升4倍
```

**后训练四阶段**：
1. 长CoT冷启动SFT（数学/代码推理能力）
2. 推理RL（规则化奖励强化探索）
3. 思维模式融合（思考+非思考统一数据集微调）
4. 通用RL（综合能力强化）

**蒸馏策略**：用旗舰模型知识蒸馏小模型（0.6B-14B），大幅降低训练成本

**关键结论**：
- Qwen3-32B-Base性能 ≈ Qwen2.5-72B-Base（参数减半性能持平）
- Qwen3 MoE仅用Qwen2.5-72B 1/3激活参数即全面超越
- Qwen3-4B可匹敌Qwen2.5-72B-Instruct

> 参考：[Qwen3 Blog](https://qwen.ai/blog?id=qwen3)、[Qwen3 Technical Report](https://raw.githubusercontent.com/QwenLM/Qwen3/main/Qwen3_Technical_Report.pdf)、[Qwen3技术报告解读](https://blog.csdn.net/wjinjie/article/details/148208426)

---

### 10.5 四大模型预训练横向对比

| 维度 | DeepSeek-V3 | DeepSeek-R1 | LLaMA 3 | Qwen3 |
|------|-------------|-------------|---------|-------|
| **发布时间** | 2024.12 | 2025.01 | 2024.04/07 | 2025.04 |
| **架构** | MoE (671B/37B激活) | 基于V3-Base | Dense Transformer | Dense + MoE双线 |
| **总参数** | 671B | 671B(基座) | 8B/70B/405B | 0.6B~235B |
| **预训练数据** | 14.8T | 复用V3基座 | 15~15.6T | **36T** |
| **语言覆盖** | 多语言 | 多语言 | 176种语言 | **119种语言/方言** |
| **核心注意力** | **MLA**（KV低秩压缩） | MLA | **GQA**（KV头=8） | **GQA + QK-Norm** |
| **MoE设计** | 256路由+共享专家, top-8 | 复用V3 | 无MoE | 128路由, top-8, 无共享专家 |
| **训练目标** | CLM + **MTP** | CLM(基座) + GRPO(RL) | CLM | CLM |
| **精度策略** | **FP8混合精度** | BF16(基座) | BF16 | BF16 |
| **上下文窗口** | 4K→128K | 复用V3 | 8K→128K | 4K→32K(推理4x→128K) |
| **数据配比** | 未详细公开 | 复用V3 | 50%通用/25%推理/17%代码/8%多语 | 实例级混合优化 |
| **特色创新** | 无辅助损失负载均衡、DualPipe | GRPO纯RL推理、顿悟时刻 | Over-training 75x、退火策略 | 思考/非思考双模式、36T超大规模 |
| **训练成本** | 2.788M H800 GPUh | ~147K H800 GPUh(仅RL) | 7.7M H100 GPUh | 未公开 |
| **训练稳定性** | 全程无不可恢复loss spike | 冷启动避免RL不稳定 | 极少loss spike | QK-Norm保证稳定性 |
| **开源协议** | MIT | MIT | Llama 3 License | **Apache 2.0** |

**四大模型的关键启示**：

1. **MoE是超大规模模型的必然选择**：DeepSeek-V3和Qwen3旗舰均采用MoE，5%左右激活率实现同性能
2. **数据规模持续突破上限**：从LLaMA 3的15T到Qwen3的36T，Over-training仍有效
3. **合成数据成为新引擎**：Qwen3用专业模型（Math/Coder）生成数万亿合成token
4. **RL是推理能力的关键路径**：DeepSeek-R1证明纯RL可催生推理，GRPO比PPO更实用
5. **FP8训练首次大规模验证**：DeepSeek-V3在671B模型上验证FP8可行，训练成本降低40%+
6. **长上下文需渐进式扩展**：所有模型都采用分阶段策略，而非一步到位
