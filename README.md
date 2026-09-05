# ML Fundamentals 学习交接文档


## 1. 当前学习节点

当前阶段：

~~~text
ML Fundamentals 基础阶段基本完成
        ↓
下一阶段：PyTorch Fundamentals
~~~

已经建立的核心训练链：

~~~text
输入 X → 模型参数 w、b → Prediction → Loss
    → Derivative / Gradient → Gradient Descent → 更新参数
~~~

已经完成或基本掌握：

- Model、Parameters、Prediction、Loss；
- MSE 与 Linear Regression from scratch；
- Derivative、Gradient、Gradient Descent、Learning Rate；
- Sigmoid、Logistic Regression、BCE 及梯度 ∂L/∂z = p-y；
- Overfitting、Generalization、Train / Validation / Test、Data Leakage；
- L1 / L2 Regularization；
- Vectorization、Matrix Shape、Batch；
- Linear Layer、MLP、ReLU 与 Activation 的作用。

尚未完成：

- PyTorch Tensor、Autograd、Computational Graph；
- nn.Module、Loss Function、Optimizer；
- 用 PyTorch 重写线性回归和逻辑回归；
- MLP 训练、评估和 Backpropagation 的逐层验证。

---

## 2. 推荐 GitHub 项目结构

~~~text
ml-fundamentals/
├── README.md
├── notes/
│   ├── 01-model-parameters-prediction-loss.md
│   ├── 02-derivative-gradient-descent.md
│   ├── 03-linear-regression.md
│   ├── 04-logistic-regression.md
│   ├── 05-generalization-regularization.md
│   ├── 06-vectorization-and-shape.md
│   └── 07-mlp-and-activation.md
├── src/
│   ├── from_scratch/
│   │   ├── linear_regression_from_scratch.py
│   │   └── logistic_regression_from_scratch.py
│   └── pytorch/
│       ├── 01_tensor_operations.py
│       ├── 02_autograd_linear_regression.py
│       ├── 03_nn_module_logistic_regression.py
│       └── 04_mlp_binary_classification.py
├── experiments/
│   └── README.md
└── requirements.txt
~~~

原始附件中的程序名称：

- Linear Regression baseline.py
- Logistic Regression from scratch.py

可以保留原名，也可以改为上面的 snake_case 文件名；改名不改变学习内容。

---

## 3. Model、Parameters、Prediction、Loss

模型是带有可学习参数的函数：

$$
\hat{y}=f(X;\theta)
$$

其中 X 是输入，ŷ 是预测值，θ 是参数。在线性模型中：

$$
\theta=(w,b), \qquad \hat{y}=wx+b
$$

多特征和 batch 的矩阵形式：

$$
Z=XW+b
$$

训练不是让参数“凭感觉变好”，而是寻找能最小化 Loss 的参数。Loss 衡量预测与真实标签之间的差异。

## 4. MSE

回归中使用的 MSE：

$$
MSE=\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)^2
$$

令 eᵢ = ŷᵢ - yᵢ，则一维线性回归的梯度为：

$$
\frac{\partial L}{\partial w}=\frac{2}{n}\sum_i e_i x_i,
\qquad
\frac{\partial L}{\partial b}=\frac{2}{n}\sum_i e_i
$$

要区分 error、per-sample loss 和对 batch 求平均后的 mean loss（scalar）。

## 5. Derivative、Gradient、Gradient Descent、Learning Rate

多个参数各自的偏导数合在一起就是 Gradient：

$$
\nabla_\theta L=
\left[
\frac{\partial L}{\partial \theta_1},
\frac{\partial L}{\partial \theta_2},
\ldots
\right]
$$

梯度指向 Loss 增长最快的方向，所以沿负梯度更新：

$$
\theta\leftarrow\theta-\eta\nabla_\theta L
$$

η 是 learning rate：

- 太小：训练慢；
- 太大：可能震荡或发散；
- 合适：Loss 通常稳定下降。

标准循环：

~~~text
初始化参数
重复 epochs：
    forward：计算 prediction
    loss：计算误差
    gradient：计算 Loss 对参数的梯度
    update：参数减去 learning rate × gradient
~~~

## 6. 已完成程序一：Linear Regression from scratch

对应原始文件：Linear Regression baseline.py。

数据：

~~~python
x = np.array([1., 2., 3., 4.])
y = np.array([5., 8., 11., 14.])
~~~

这组数据大致服从 y = 3x + 2。核心训练逻辑：

~~~python
y_hat = w * x + b
error = y_hat - y
loss = np.mean(error ** 2)

gradient_w = 2 * error @ x.T / len(x)
gradient_b = np.mean(2 * error)

w = w - lr * gradient_w
b = b - lr * gradient_b
~~~

它体现了 Forward、MSE、Gradient、Gradient Descent 和训练观察。当前一维数组场景中：

~~~python
2 * error @ x.T / len(x)
np.mean(2 * error * x)
~~~

表达同一类梯度计算。

这份程序是理解训练机制的最小 baseline，不是完整生产建模流程；目前没有覆盖多特征矩阵、数据划分、标准化、正则化和真实数据集泛化评估。

## 7. Logistic Regression、Sigmoid、BCE

二分类数据流：

$$
x\rightarrow z=wx+b\rightarrow p=\sigma(z)\rightarrow BCE
$$

Sigmoid：

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

它把实数映射到 (0,1)，可解释为正类概率。概率不是最终类别，需要通过阈值转换：

~~~python
y_pred = (p >= 0.5).astype(int)
~~~

BCE：

$$
BCE=-\frac{1}{n}\sum_i
[y_i\log p_i+(1-y_i)\log(1-p_i)]
$$

关键梯度结论：

$$
\frac{\partial L}{\partial z}=p-y
$$

因此：

$$
\frac{\partial L}{\partial w}
=\frac{1}{n}\sum_i(p_i-y_i)x_i,
\qquad
\frac{\partial L}{\partial b}
=\frac{1}{n}\sum_i(p_i-y_i)
$$

它来自链式法则。Backpropagation 的本质，就是高效地把 Chain Rule 从输出端向输入端计算。

## 8. 已完成程序二：Logistic Regression from scratch

对应原始文件：Logistic Regression from scratch.py。

数据：

~~~python
x = np.array([1., 2., 3., 4., 5., 6.])
y = np.array([0., 0., 0., 1., 1., 1.])
~~~

核心训练逻辑：

~~~python
z = w * x + b
p = 1 / (1 + np.exp(-z))
loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

gradient_w = (p - y) @ x.T / len(x)
gradient_b = np.mean(p - y)

w = w - lr * gradient_w
b = b - lr * gradient_b
~~~

训练结束后重新 forward，并输出 w、b、probability、prediction 和 true label。

必须区分：

~~~text
score z ≠ probability p ≠ class prediction y_pred
~~~

后续改进：对 p 做数值稳定处理；改写为矩阵版本；记录 Loss 曲线；用 validation set 选择超参数或阈值；报告 precision、recall、F1、ROC-AUC、PR-AUC；在类别不平衡问题中关注正类识别和业务成本。

## 9. Overfitting、Generalization、Train / Validation / Test

过拟合是模型不仅学习规律，还记住训练数据噪声或偶然细节：

~~~text
Training performance 很好
但未见数据 performance 很差
~~~

真正关心的是未来未见数据的泛化能力。

| 数据集 | 作用 | 是否参与参数训练 |
|---|---|---|
| Train | 拟合模型参数 | 是 |
| Validation | 选择模型、超参数、阈值 | 否，通常用于选择 |
| Test | 最终一次性估计泛化表现 | 否 |

测试集不能被反复用来调参。常见数据泄露包括：用全体数据计算均值、标准差或频率；用测试集结果反复选模型；使用预测时点之后才产生的字段；把目标变量或其变形放进特征。

原则：训练阶段学习到的 preprocessing、feature mapping 和 model parameters，都应该只使用训练数据拟合。

## 10. L1 / L2 Regularization

总目标：

$$
L_{total}=L_{data}+\lambda L_{reg}
$$

λ 控制拟合数据和控制复杂度之间的权衡。

L2：

$$
L_{L2}=\lambda\sum_j w_j^2,
\qquad
\frac{\partial}{\partial w_j}\lambda w_j^2=2\lambda w_j
$$

L2 倾向于让权重整体变小，但不一定精确为 0。

L1：

$$
L_{L1}=\lambda\sum_j|w_j|
$$

对于非零参数：

$$
\frac{\partial}{\partial w_j}\lambda|w_j|
=\lambda\operatorname{sign}(w_j)
$$

L1 更容易产生稀疏参数，也可能产生类似 Feature Selection 的效果。准确说法是：非零参数的 L1 penalty 梯度绝对值为 λ；零点使用次梯度理解。

λ = 0 只表示没有这项正则化约束，不代表一定过拟合；λ 很大则可能导致 underfitting。Bias 也是参数，但 weights 更直接体现模型对 features 的敏感程度，bias 更多承担整体平移作用，所以很多实现主要正则化 weights。

## 11. Vectorization、Matrix Shape、Batch

如果 X.shape = (4,3)，表示 4 个 samples、每个 sample 有 3 个 features：

~~~text
行 = sample
列 = feature
~~~

xᵢⱼ 是第 i 个样本的第 j 个特征。

若 X.shape = (4,3)，w.shape = (3,1)：

$$
Xw:(4,3)(3,1)\rightarrow(4,1)
$$

第一个样本：

$$
z_1=w_1x_{11}+w_2x_{12}+w_3x_{13}+b
$$

不能把同一个样本的特征索引写成 x₁₁、x₂₂、x₃₃。

若 X.shape = (32,10)：

~~~text
X       : (32, 10)
w       : (10, 1)
z = Xw+b: (32, 1)
p       : (32, 1)
~~~

32 是 batch size，不是参数维度。每个样本的 loss 可以是 (32,1)，对 batch 求 mean 后训练 Loss 通常是 scalar，即 shape 为 ()。

对于 X.shape = (128,50)：

~~~text
Linear(50, 20) → (128, 20)
Linear(20, 5)  → (128, 5)
Linear(5, 1)   → (128, 1)
~~~

128 是同一个 batch 中的样本数；变化的是 representation dimension。特征维度不一定只能变小，50 → 128 → 512 → 64 → 1 也可以。

## 12. Linear Layer、MLP、ReLU、Activation

Linear Layer 的核心：

$$
Z=XW+b
$$

输入 (64,100)、输出 (64,32) 时：

$$
(64,100)(100,32)\rightarrow(64,32)
$$

对应：

~~~python
nn.Linear(100, 32)
~~~

PyTorch 内部的 layer.weight.shape 通常是 (32,100)，因为 forward 使用等价的转置形式；数学上仍可按 (100,32) 理解乘法。

如果没有 Activation：

$$
h=XW_1+b_1,\qquad z=hW_2+b_2
$$

代入后：

$$
z=XW_1W_2+b_1W_2+b_2=XW'+b'
$$

所以多层 Linear 仍可合并为一个仿射变换。Activation 引入非线性，才让网络获得更强的表达能力。

ReLU：

$$
ReLU(z)=\max(0,z)
$$

~~~text
[-3, -1, 0, 2, 5] → [0, 0, 0, 2, 5]
~~~

它是 element-wise operation，shape 不变。

典型 MLP：

~~~python
nn.Linear(50, 20)
nn.ReLU()
nn.Linear(20, 5)
nn.ReLU()
nn.Linear(5, 1)
~~~

如果 X.shape = (128,50)，数据流是 (128,50) → (128,20) → (128,5) → (128,1)。实际 PyTorch 二分类训练优先使用 BCEWithLogitsLoss，最后一层输出 logits，不在模型中额外 Sigmoid。

## 13. 已纠正的易混淆点

1. 多层 Linear 没有 Activation 时仍是一个仿射变换。
2. Batch size 不是 weight 维度；X=(32,10) 时 w=(10,1)。
3. p 是概率，不是 0/1 标签；阈值后才是 y_pred。
4. BCE 对 batch 求 mean 后通常是 scalar。
5. 第一个样本的特征必须来自第一行。
6. Training Loss 很低不代表泛化好。
7. λ=0 只表示没有正则化约束。
8. Bias 不正则化不是因为它绝对不会导致过拟合。
9. L1 的固定梯度大小只适用于非零参数。
10. 必须区分 logit、probability 和 class label。

## 14. 下一阶段：PyTorch Fundamentals

学习顺序：

~~~text
NumPy Array → Tensor → requires_grad → Computational Graph
→ backward() → Gradient → nn.Module → Loss → Optimizer
→ MLP → Backpropagation
~~~

### Tensor

~~~python
import torch

x = torch.tensor([[1., 2.], [3., 4.]])
print(x.shape)
print(x.dtype)
print(x @ x.T)
~~~

掌握 shape、dtype、indexing、element-wise operation、矩阵乘法，以及 @ 与 * 的区别。

### Autograd

把手写的：

~~~python
w = w - lr * gradient_w
~~~

逐步换成：

~~~python
loss.backward()
with torch.no_grad():
    w -= lr * w.grad
~~~

重点是 requires_grad=True、Computational Graph、backward()、grad、torch.no_grad() 和每轮清空旧梯度。

### nn.Module

~~~python
class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)
~~~

理解 Module 管理参数，model.parameters() 返回可学习参数，forward() 定义数据流。

### Loss 与 Optimizer

- nn.MSELoss()：回归；
- nn.BCELoss()：接收概率；
- nn.BCEWithLogitsLoss()：接收 logits，通常更稳定。

训练顺序：

~~~python
optimizer.zero_grad()
prediction = model(x)
loss = loss_fn(prediction, y)
loss.backward()
optimizer.step()
~~~

| 手写版本 | PyTorch |
|---|---|
| 计算 prediction | model(x) |
| 计算 loss | loss_fn(prediction, y) |
| 手推 gradient | loss.backward() |
| 参数更新 | optimizer.step() |
| 清空旧梯度 | optimizer.zero_grad() |

### MLP 与 Backpropagation

~~~python
class MLP(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(in_features, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1),
        )

    def forward(self, x):
        return self.network(x)
~~~

若使用 BCEWithLogitsLoss，最后一层直接输出 logits，不要额外 Sigmoid。

Backpropagation 的理解顺序：

1. forward 建立计算图；
2. loss 是图的终点；
3. backward 从终点应用 Chain Rule；
4. 每个参数获得自己的梯度；
5. optimizer 根据梯度更新参数。

## 15. 后续会话交接规则

1. 先把 NumPy 手写版本和 PyTorch 版本逐行对应。
2. 每次引入代码，都说明输入、输出、参数和 Loss 的 shape。
3. 新数学概念先说明它在训练流程中的位置，再给公式。
4. 不要把 toy dataset 的结果写成真实项目性能。
5. 优先让学习者自己写核心代码，再检查和纠错。
6. 保留 batch、feature、probability、threshold、scalar loss 和矩阵 shape 的纠正。
7. 真实数据项目必须明确 Train / Validation / Test，并检查 preprocessing 是否泄露。
8. PyTorch 先完成 Tensor → Autograd → nn.Module → Loss → Optimizer，再开始 MLP。

## 16. 当前完成度

### 已掌握或基本掌握

- [x] Model、Parameters、Prediction、Loss
- [x] MSE
- [x] Derivative、Gradient、Gradient Descent、Learning Rate
- [x] Linear Regression from scratch
- [x] Sigmoid、Logistic Regression、BCE
- [x] ∂L/∂z = p-y
- [x] Overfitting、Generalization、Train / Validation / Test、Data Leakage
- [x] L1 / L2 Regularization
- [x] Vectorization、Matrix Shape、Batch
- [x] Linear Layer、MLP、ReLU、Activation

### 待完成

- [ ] Tensor 基础
- [ ] Autograd 梯度验证
- [ ] PyTorch Linear Regression
- [ ] PyTorch Logistic Regression
- [ ] nn.Module、Loss、Optimizer
- [ ] MLP 二分类
- [ ] Backpropagation 逐层理解
- [ ] DataLoader、mini-batch、evaluation mode
- [ ] 真实数据集上的可复现实验

## 17. 一句话总结

目前已经从“只会调用模型”推进到能够解释并手写：

~~~text
Model → Prediction → Loss → Gradient → Parameter Update
~~~

下一步是把这套手写机制映射到：

~~~text
PyTorch Tensor → Autograd → nn.Module → Loss → Optimizer → MLP
~~~

这份 README 是 ML Fundamentals 到 PyTorch Fundamentals 的交接点，不代表 PyTorch 阶段已经完成。

## 18. 更新记录

| 日期 | 更新内容 |
|---|---|
| 2026-09 | 整理 ML Fundamentals、两个 NumPy from-scratch 程序、易错点和 PyTorch 学习路线 |


