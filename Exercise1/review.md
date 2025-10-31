1.n*n单位矩阵：
np.eye(n)

2.线性回归的代价函数：
$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2
$$
J = 1 / (2 * m) * np.sum((X.dot(theta) - y) ** 2)

3.梯度下降求解线性回归：
$$
\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}
$$

$ grad= \frac{1}{m} X^T (X \theta - y) $

for i in range(num_iters):
        # 计算梯度
        grad = (1/m) * X.T @ (X @ theta - y)
        
        # 更新 theta
        theta = theta - alpha * grad
        
        # 在每次迭代中保存代价 J
        J_history.append(computeCost(X, y, theta))
    
    return theta, J_history

4.归一化：
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])

    # 计算每个特征的均值
    mu = np.mean(X, axis=0)
    
    # 计算每个特征的标准差
    sigma = np.std(X, axis=0)
    
    # 归一化每个特征
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma

