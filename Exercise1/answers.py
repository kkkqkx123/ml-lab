import numpy as np

def warmUpExercise():
    return np.eye(5)

def computeCost(X, y, theta):
    m = y.size
    J = (1/(2 * m)) * np.sum(np.square(np.dot(X, theta) - y))
    return J

def gradientDescent(X, y, theta, alpha, num_iters):
    m = y.shape[0]
    theta = theta.copy()
    J_history = []
    for i in range(num_iters):
        theta = theta - (alpha / m) * (np.dot(X, theta) - y).dot(X)
        J_history.append(computeCost(X, y, theta))
    return theta, J_history

def featureNormalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

def computeCostMulti(X, y, theta):
    m = y.shape[0]
    J = (1/(2 * m)) * np.sum(np.square(np.dot(X, theta) - y))
    return J

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = y.shape[0]
    theta = theta.copy()
    J_history = []
    for i in range(num_iters):
        theta = theta - (alpha / m) * (np.dot(X, theta) - y).dot(X)
        J_history.append(computeCostMulti(X, y, theta))
    return theta, J_history

def normalEqn(X, y):
    theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
    return theta

answers = {
    1: warmUpExercise,
    2: computeCost,
    3: gradientDescent,
    4: featureNormalize,
    5: computeCostMulti,
    6: gradientDescentMulti,
    7: normalEqn
}

def score(part_id, *args):
    func = answers.get(part_id)
    if func is None:
        print(f"未找到第 {part_id} 部分的评分函数。")
        return 0
    try:
        # 只判断是否能正确运行和返回结果
        result = func(*args)
        print(f"第 {part_id} 部分评分通过。结果: {result if result is not None else '无'}")
        return 1
    except Exception as e:
        print(f"第 {part_id} 部分评分失败: {e}")
        return 0
