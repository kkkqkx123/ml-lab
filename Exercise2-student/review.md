# Exercise2 代码解释

## 1. Sigmoid函数

```python
# 计算 Sigmoid 函数值
g = 1 / (1 + np.exp(-z))
```

**解释：**
- `np.exp(-z)`：计算e的-z次方
- `1 + np.exp(-z)`：分母部分
- `1 / (1 + np.exp(-z))`：完整的Sigmoid函数公式
- 该函数将任意实数映射到(0,1)区间，用于逻辑回归的概率计算

## 2. 逻辑回归代价函数和梯度

```python
# 计算 Sigmoid 函数值
h = sigmoid(X.dot(theta))

# 计算代价函数
J = -1/m * (y.dot(np.log(h)) + (1-y).dot(np.log(1-h)))

# 计算梯度
grad = 1/m * X.T.dot(h - y)
```

**代价函数解释：**
- `h = sigmoid(X.dot(theta))`：计算假设函数值，即预测概率
- `y.dot(np.log(h))`：正例部分的代价，当y=1时起作用
- `(1-y).dot(np.log(1-h))`：负例部分的代价，当y=0时起作用
- `-1/m * (...)`：平均代价，取负号是因为对数函数在(0,1)区间为负值

**梯度解释：**
- `h - y`：预测值与真实值的差异
- `X.T.dot(h - y)`：特征矩阵转置与误差向量的点积，得到每个特征的梯度分量
- `1/m * ...`：平均梯度

**数学公式：**
$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1-y^{(i)}) \log(1-h_\theta(x^{(i)})) \right]
$$

$$
\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}
$$

## 3. 预测函数

```python
# 计算 Sigmoid 函数值
h = sigmoid(X.dot(theta))

# 计算预测值
p = h >= 0.5
```

**解释：**
- `h = sigmoid(X.dot(theta))`：计算每个样本的预测概率
- `p = h >= 0.5`：将概率转换为二元分类结果
  - 如果概率 ≥ 0.5，预测为1（正例）
  - 如果概率 < 0.5，预测为0（负例）
- 0.5是常用的决策边界阈值

## 4. 正则化逻辑回归代价函数和梯度

```python
# 计算 Sigmoid 函数值
h = sigmoid(X.dot(theta))

# 计算代价函数（不含正则化项）
J = -1/m * (y.dot(np.log(h)) + (1-y).dot(np.log(1-h)))

# 添加正则化项（注意不对theta[0]正则化）
J += lambda_/(2*m) * np.sum(theta[1:]**2)

# 计算梯度（不含正则化项）
grad = 1/m * X.T.dot(h - y)

# 添加正则化项到梯度（注意不对theta[0]正则化）
grad[1:] += lambda_/m * theta[1:]
```

**正则化代价函数解释：**
- 第一行：计算基本逻辑回归代价
- 第二行：添加L2正则化项，惩罚大的参数值
- `theta[1:]`：排除截距项theta[0]，不对其进行正则化
- `lambda_/(2*m)`：正则化系数除以2m

**正则化梯度解释：**
- 第一行：计算基本梯度
- 第二行：对非截距参数添加正则化梯度项
- `grad[1:]`：只对非截距参数的梯度进行修改
- `lambda_/m * theta[1:]`：正则化梯度项

**数学公式：**
$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1-y^{(i)}) \log(1-h_\theta(x^{(i)})) \right] + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2
$$

$$
\frac{\partial J(\theta)}{\partial \theta_0} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_0^{(i)}
$$

$$
\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)} + \frac{\lambda}{m} \theta_j \quad (j \geq 1)
$$

## 关键概念总结

1. **Sigmoid函数**：将线性组合转换为概率值
2. **逻辑回归代价函数**：衡量预测概率与真实标签的差异
3. **梯度计算**：指导参数优化的方向
4. **正则化**：防止过拟合，提高模型泛化能力
5. **预测阈值**：0.5作为默认的二元分类决策边界