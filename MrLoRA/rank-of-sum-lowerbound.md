矩阵和 $\operatorname{rank}\left(\sum_{i=1}^k A_i\right)$ 的**通用下界**由**秩的三角不等式**（reverse triangle inequality）给出：

\[
\boxed{\operatorname{rank}\left(\sum_{i=1}^k A_i\right) \geq \max_{1 \leq j \leq k} \left\{ \operatorname{rank}(A_j) - \sum_{\substack{i=1 \\ i \neq j}}^k \operatorname{rank}(A_i) \right\}}
\]

或等价地写成：

\[
\operatorname{rank}\left(\sum_{i=1}^k A_i\right) \geq \max_{j} \operatorname{rank}(A_j) - \sum_{i \neq j} \operatorname{rank}(A_i) = 2\cdot\max_j \operatorname{rank}(A_j) - \sum_{i=1}^k \operatorname{rank}(A_i)
\]

---

### 推导思路

该下界源于秩的基本不等式 $\operatorname{rank}(X+Y) \geq \bigl|\operatorname{rank}(X) - \operatorname{rank}(Y)\bigr|$，即**反向三角不等式**。

对任意固定的 $j$，将和式拆分为 $A_j$ 与 $\sum_{i\neq j} A_i$：

\[
\operatorname{rank}\left(\sum_{i=1}^k A_i\right) = \operatorname{rank}\left(A_j + \sum_{i\neq j} A_i\right) \geq \operatorname{rank}(A_j) - \operatorname{rank}\left(\sum_{i\neq j} A_i\right)
\]

再对上式右边的和式递归应用上界 $\operatorname{rank}(\sum_{i\neq j} A_i) \leq \sum_{i\neq j} \operatorname{rank}(A_i)$，即得：

\[
\operatorname{rank}\left(\sum_{i=1}^k A_i\right) \geq \operatorname{rank}(A_j) - \sum_{i\neq j} \operatorname{rank}(A_i)
\]

由于该式对任意 $j$ 都成立，取最大值即得到最终下界。

---

### 特殊情况

- **当 $k=2$ 时**：退化为经典形式
  \[
  \operatorname{rank}(A_1 + A_2) \geq \bigl|\operatorname{rank}(A_1) - \operatorname{rank}(A_2)\bigr|
  \]
  
- **当下界为非正数时**：秩始终非负，因此实际有效下界为 $\max\{0, \text{上述表达式}\}$。当下界 $\leq 0$ 时，该不等式不提供有效信息（因为秩本身就 $\geq 0$）。

- **紧致性（Tightness）**：该下界是**紧的**（可以达到）。例如取 $A_2 = -A_1$，则 $\operatorname{rank}(A_1+A_2)=0$，而 $\bigl|\operatorname{rank}(A_1)-\operatorname{rank}(A_2)\bigr|=0$。

---

### 补充：涉及像空间（Image）的下界

若引入矩阵的像空间（列空间）$\operatorname{Im}(A)$，则有更精细但复杂的下界：

\[
\operatorname{rank}\left(\sum_{i=1}^k A_i\right) \geq \sum_{i=1}^k \operatorname{rank}(A_i) - \sum_{1\leq i<j\leq k} \dim\bigl(\operatorname{Im}(A_i) \cap \operatorname{Im}(A_j)\bigr) \cdot \mathbb{I}_{\text{依赖关系}}
\]

但在大多数应用场景中，前述基于秩相减的简单下界已足够实用。