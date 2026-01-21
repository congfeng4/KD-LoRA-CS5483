Hugging Face 的 `peft` 库目前已经成为了参数高效微调的“瑞士军刀”，除了你提到的 **AdaLoRA**，它还支持非常多针对不同场景优化的 LoRA 变体。

根据最新的文档和实现，可以将这些变体分为以下几类：

### 1. 性能增强类（更准、更稳）

* **DoRA (Weight-Decomposed Low-Rank Adaptation)**:
* **原理**：将权重分解为“幅值 (Magnitude)”和“方向 (Direction)”，仅对方向部分应用 LoRA。
* **优点**：训练能力最接近全量微调（Full Fine-tuning），在很多任务上超越了标准 LoRA。


* **LoRA+**:
* **原理**：为 LoRA 的 A 矩阵和 B 矩阵设置不同的学习率（通常 B 的学习率远大于 A）。
* **优点**：解决了大模型在微调时由于初始化导致的数据效率低下的问题，收敛更快。


* **rsLoRA (Rank-Stabilized LoRA)**:
* **原理**：通过引入一个学习率缩放因子  来稳定训练。
* **优点**：允许你设置非常大的 Rank（如 ）而不会导致模型训练崩溃。



### 2. 参数压缩与效率类（更轻、更省）

* **QLoRA (Quantized LoRA)**:
* **原理**：将底座模型量化为 4-bit (NF4)，然后在上面跑 LoRA。
* **优点**：目前最流行的变体。让你能在 24G 显存的卡上微调 70B 的模型。


* **LoHa (Low-Rank Hadamard Product)**:
* **原理**：使用哈达玛积（元素相乘）而非矩阵相乘来组合低秩矩阵。
* **优点**：在相同的参数量下具有更高的表达能力（Rank 的上界更高），常用于 Stable Diffusion 微调。


* **LoKr (Low-Rank Kronecker Product)**:
* **原理**：利用克罗内克积将大矩阵分解为极小的块。
* **优点**：极度节省参数，适合超大规模参数层的压缩。


* **VeRA (Vector-based Random Aggregation)**:
* **原理**：冻结大部分随机生成的 LoRA 矩阵，只训练微小的缩放向量。
* **优点**：参数量比标准 LoRA 还要少 10 倍以上。



### 3. 初始化与算法改进类

* **PiSSA (Principal Singular values and Singular vectors Adaptation)**:
* **原理**：使用 SVD 对原始权重进行分解，将主要的奇异值作为 LoRA 初值。
* **优点**：比 LoRA 初始化更科学，收敛速度极快。


* **BOFT (Block-Orthogonal Fine-Tuning)**:
* **原理**：利用块正交矩阵进行微调。
* **优点**：能更好地保持预训练模型的原始分布，减少灾难性遗忘。



---

### 如何在代码中切换？

在 `peft` 中切换这些变体通常非常简单，只需修改 `Config` 中的参数。例如：

```python
# 使用 DoRA
config = LoraConfig(
    use_dora=True,  # 只需开启这个开关
    r=8, 
    target_modules=["q_proj", "v_proj"]
)

# 使用 rsLoRA
config = LoraConfig(
    use_rslora=True, # 开启 rank 稳定缩放
    r=256,           # 可以尝试极高的 rank
    target_modules=["q_proj", "v_proj"]
)

```

### 总结建议

* **如果你追求极致精度**：试试 **DoRA**。
* **如果你显存极度紧张**：选 **QLoRA**。
* **如果你在处理图像/多模态**：**LoHa** 或 **LoKr** 往往效果更好。
* **如果你在做极小数据集（如 WNLI）**：**AdaLoRA** 或 **PiSSA** 能有效防止模型退化。

**既然你手头有 WNLI 的模型，想试试用 DoRA 重新跑一次看看准确率是否有提升吗？**
