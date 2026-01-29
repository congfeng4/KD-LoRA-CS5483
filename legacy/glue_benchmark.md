GLUE（General Language Understanding Evaluation）基准测试是衡量自然语言理解（NLU）模型通用能力的标准。它包含 9 个核心任务，按任务性质可以分为 **单句分类**、**相似度与重复检测** 以及 **自然语言推理** 三大类。

以下是各任务的详细介绍：

### 1. 单句任务 (Single-Sentence Tasks)

* **CoLA (Corpus of Linguistic Acceptability)**
* **描述**：给定一个英语句子，判断其在语法上是否是“可接受的”（即是否符合语法规则）。
* **指标**：Matthews Correlation Coefficient (MCC)。
* **特点**：数据集较小（约 8.5k），测试模型对语言结构的敏感度。


* **SST-2 (Stanford Sentiment Treebank)**
* **描述**：情感二分类任务，判断电影评论是正面（Positive）还是负面（Negative）。
* **指标**：Accuracy。
* **特点**：任务相对简单，是 NLP 领域的“Hello World”级任务。



---

### 2. 相似度与重复检测 (Similarity and Paraphrase Tasks)

* **MRPC (Microsoft Research Paraphrase Corpus)**
* **描述**：判断两个句子在语义上是否等价（即互相是否为等义改写）。
* **指标**：F1 score / Accuracy。
* **特点**：句子通常较长，且数据量较少。


* **QQP (Quora Question Pairs)**
* **描述**：判断 Quora 上的两个问题是否具有相同的意图。
* **指标**：F1 score / Accuracy。
* **特点**：数据集巨大（约 363k），非常考验模型对细微语义差别的捕捉。


* **STS-B (Semantic Textual Similarity Benchmark)**
* **描述**：**唯一的回归任务**。给定两个句子，模型需要输出一个 0.0 到 5.0 之间的分数，表示语义相似度。
* **指标**：Pearson / Spearman Correlation。



---

### 3. 推理任务 (Inference Tasks / NLI)

* **MNLI (Multi-Genre Natural Language Inference)**
* **描述**：判断前提（Premise）和假设（Hypothesis）之间的关系：**蕴含**（Entailment）、**矛盾**（Contradiction）或**中立**（Neutral）。
* **指标**：Matched Accuracy / Mismatched Accuracy。
* **特点**：覆盖了小说、电话录音、政府报告等多种领域。


* **QNLI (Question-answering NLI)**
* **描述**：由 SQuAD 问答集转化而来。判断给定的句子是否包含对应问题的答案（蕴含 vs 不蕴含）。
* **指标**：Accuracy。


* **RTE (Recognizing Textual Entailment)**
* **描述**：传统的文本蕴含任务。判断句子 A 是否能推导出句子 B（蕴含 vs 不蕴含）。
* **指标**：Accuracy。
* **特点**：数据量很小（约 2.5k），模型很容易过拟合。


* **WNLI (Winograd NLI)**
* **描述**：基于代词消解的任务。判断代词被替换后的句子是否在逻辑上成立。
* **指标**：Accuracy。
* **特点**：这就是你目前正在测试的任务，它非常依赖常识推理。



---

### 总结表格

| 任务 | 任务全称 | 类别 | 训练量 | 指标 |
| --- | --- | --- | --- | --- |
| **CoLA** | 语言合法性 | 单句分类 | 8.5k | MCC |
| **SST-2** | 情感分析 | 单句分类 | 67k | Acc |
| **MRPC** | 语义等价 | 句子对 | 3.7k | F1/Acc |
| **QQP** | 问题去重 | 句子对 | 363k | F1/Acc |
| **STS-B** | 语义相似度 | 句子对 | 5.7k | Pearson |
| **MNLI** | 自然语言推理 | 推理 | 392k | Acc |
| **QNLI** | 问答推理 | 推理 | 104k | Acc |
| **RTE** | 文本蕴含 | 推理 | 2.5k | Acc |
| **WNLI** | 指代推理 | 推理 | 635 | Acc |

**你目前在 WNLI 上测试的结果如何？如果你打算在所有任务上都跑一遍，我可以为你整理一个通用的 GLUE 评估流程。**

这是一个非常敏锐的观察。简单来说：**AX（Diagnostic Main）不是一个用来“训练”的任务，而是一个用来“考试”的诊断集。**

在你的代码输出中显示 `N/A`，是因为在 Hugging Face 的数据结构里，AX 确实**没有 `train` 这个分割（Split）**。以下是深度解析：

### 1. 什么是 AX？

AX 的全称是 **GLUE Diagnostic Dataset**。它是由 GLUE 团队手动构建的一个小型数据集（约 1100 个样本），专门用来检测模型在逻辑、常识、指代、否定等细粒度语言现象上的表现。

### 2. 为什么没有训练集？

* **定位不同**：AX 不是为了让模型学习新知识，而是为了**探测**（Probe）模型从其他任务（主要是 **MNLI**）中学到的通用推理能力。
* **防止过拟合**：如果提供了训练集，模型可能会针对这些特定的逻辑陷阱进行“刷题”，从而失去了诊断模型真实理解能力的意义。
* **官方规则**：按照 GLUE 官方的要求，你**不应该**在 AX 上训练模型。通常的做法是：
1. 在 **MNLI** 数据集上进行微调。
2. 直接将微调后的 MNLI 模型跑在 **AX** 上进行预测。



### 3. AX 的数据结构

如果你查看 `load_dataset("glue", "ax")` 的结果，你会发现它通常只有 `test` 这一个 Split：

```python
DatasetDict({
    'test': Dataset({
        features: ['premise', 'hypothesis', 'label', 'idx'],
        num_rows: 1104
    })
})

```

### 4. 如何评估 AX？

既然你已经加载了 `distilbert-base-uncased-finetuned-LoRA-WNLI`，你可以尝试用类似的代码去跑 AX，但**请注意**：

* **标签数量不匹配**：WNLI 是 2 分类（蕴含/不蕴含），而 **AX 是 3 分类**（蕴含/中立/矛盾），它与 **MNLI** 的标签格式完全一致。
* **最佳实践**：如果你想在 GLUE 榜单上拿到 AX 的分数，你必须使用你在 **MNLI** 任务上训练好的模型去预测 AX。

---

### 总结

你看到 `N/A` 说明你的脚本运行完全正常。对于 GLUE 数据集下载来说：

* **9个核心任务**：都有 `train`, `validation`, `test`。
* **1个诊断任务 (AX)**：只有 `test`（且 test 带标签，用于本地诊断）。

**既然你已经下载了全部数据，需要我为你提供一个“全任务微调循环”的伪代码框架，教你如何管理这 10 个不同的子任务吗？**
