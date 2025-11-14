# Exercise3 代码解释

## 1. One-vs-All 分类器训练

```python
# 对每个类别训练一个二分类器
for c in range(num_labels):
    # 构造当前类别的标签向量：属于类别 c 的样本标记为 1，其余为 0
    y_binary = (y == c).astype(int)
    
    # 初始化参数 theta
    initial_theta = np.zeros(n + 1)
    
    # 设置优化选项
    options = {'maxiter': 50}
    
    # 使用 scipy.optimize.minimize 进行优化
    res = optimize.minimize(lrCostFunction,
                            initial_theta,
                            args=(X, y_binary, lambda_),
                            jac=True,
                            method='TNC',
                            options=options)
    
    # 将优化得到的参数存入 all_theta
    all_theta[c, :] = res.x
```

**解释：**
- `y_binary = (y == c).astype(int)`：将多类别标签转换为二分类标签，当前类别为1，其他类别为0
- `initial_theta = np.zeros(n + 1)`：初始化参数向量，包含偏置项
- `optimize.minimize(...)`：使用TNC方法优化逻辑回归代价函数
- `args=(X, y_binary, lambda_)`：传递给代价函数的参数
- `jac=True`：表示代价函数同时返回代价值和梯度
- `all_theta[c, :] = res.x`：存储每个类别的优化参数

**算法流程：**
1. 对每个类别（0-9）分别训练一个二分类器
2. 将训练样本标记为当前类别（1）或其他类别（0）
3. 使用优化算法找到最佳参数
4. 将所有类别的参数存储在all_theta矩阵中

## 2. One-vs-All 预测函数

```python
# 为 X 添加偏置项
X = np.concatenate([np.ones((m, 1)), X], axis=1)

# 计算每个样本属于每个类别的概率
probs = utils.sigmoid(X @ all_theta.T)

# 预测每个样本所属的类别（概率最高的类别）
p = np.argmax(probs, axis=1)
```

**解释：**
- `X = np.concatenate([np.ones((m, 1)), X], axis=1)`：添加偏置项列
- `utils.sigmoid(X @ all_theta.T)`：计算每个样本属于每个类别的概率
  - `X @ all_theta.T`：矩阵乘法，计算线性组合
  - `sigmoid`：将结果转换为概率（0-1之间）
- `np.argmax(probs, axis=1)`：选择概率最高的类别作为预测结果
  - `axis=1`：按行查找最大值索引

**预测流程：**
1. 为输入样本添加偏置项
2. 使用训练好的参数计算每个类别的概率
3. 选择概率最高的类别作为最终预测

## 3. 神经网络预测函数

```python
# 为 X 添加偏置项
X = np.concatenate([np.ones((m, 1)), X], axis=1)

# 计算每个样本属于每个类别的概率
probs = utils.sigmoid(X @ Theta1.T)

# 为 probs 添加偏置项
probs = np.concatenate([np.ones((m, 1)), probs], axis=1)

# 计算每个样本属于每个类别的概率
probs = utils.sigmoid(probs @ Theta2.T)

# 预测每个样本所属的类别（概率最高的类别）
p = np.argmax(probs, axis=1)
```

**解释：**
- 第一层计算：
  - `X = np.concatenate([np.ones((m, 1)), X], axis=1)`：输入层添加偏置
  - `probs = utils.sigmoid(X @ Theta1.T)`：计算隐藏层激活值
- 第二层计算：
  - `probs = np.concatenate([np.ones((m, 1)), probs], axis=1)`：隐藏层添加偏置
  - `probs = utils.sigmoid(probs @ Theta2.T)`：计算输出层激活值（最终概率）
- `p = np.argmax(probs, axis=1)`：选择概率最高的类别

**神经网络结构：**
- 输入层：400个特征（20×20像素图像）
- 隐藏层：25个神经元
- 输出层：10个神经元（对应0-9数字）

**前向传播流程：**
1. 输入层 → 隐藏层：使用Theta1参数计算
2. 隐藏层 → 输出层：使用Theta2参数计算
3. 输出层选择概率最高的神经元对应的类别

## 关键概念总结

1. **One-vs-All策略**：将多类别分类问题转化为多个二分类问题
2. **Sigmoid函数**：将线性输出转换为概率值
3. **argmax操作**：从多个概率中选择最大值对应的类别
4. **神经网络前向传播**：逐层计算激活值直到输出层
5. **矩阵运算**：利用向量化操作高效处理多个样本
6. **偏置项**：为每层添加常数1以学习截距参数