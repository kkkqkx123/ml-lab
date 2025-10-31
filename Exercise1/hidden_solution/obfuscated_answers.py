"""
混淆后的答案模块 - 使用代码混淆技术
通过变量名混淆、控制流混淆等方式增加阅读难度
"""
import numpy as np
import types

# 混淆工具函数
def _mangle_name(name):
    """简单的名字混淆函数"""
    return '_' + ''.join(chr(ord(c) + 1) if c.isalpha() else c for c in name)

def _create_obfuscated_function(func_code, func_name):
    """创建混淆的函数"""
    # 创建新的函数对象
    code_obj = compile(func_code, '<string>', 'exec')
    func_dict = {}
    exec(code_obj, {'np': np}, func_dict)
    return func_dict[func_name]

# 混淆后的函数实现
_OBFUSCATED_CODE = {
    'warmUpExercise': """
def _a1():
    return np.eye(5)
""",
    
    'computeCost': """
def _b2(X, y, theta):
    m = y.size
    J = (1/(2 * m)) * np.sum(np.square(np.dot(X, theta) - y))
    return J
""",
    
    'gradientDescent': """
def _c3(X, y, theta, alpha, num_iters):
    m = y.shape[0]
    theta = theta.copy()
    J_history = []
    for i in range(num_iters):
        theta = theta - (alpha / m) * (np.dot(X, theta) - y).dot(X)
        J_history.append(_b2(X, y, theta))
    return theta, J_history
""",
    
    'featureNormalize': """
def _d4(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma
""",
    
    'computeCostMulti': """
def _e5(X, y, theta):
    m = y.shape[0]
    J = (1/(2 * m)) * np.sum(np.square(np.dot(X, theta) - y))
    return J
""",
    
    'gradientDescentMulti': """
def _f6(X, y, theta, alpha, num_iters):
    m = y.shape[0]
    theta = theta.copy()
    J_history = []
    for i in range(num_iters):
        theta = theta - (alpha / m) * (np.dot(X, theta) - y).dot(X)
        J_history.append(_e5(X, y, theta))
    return theta, J_history
""",
    
    'normalEqn': """
def _g7(X, y):
    theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
    return theta
"""
}

# 创建混淆的函数
_func_mapping = {}
_func_names = {
    '_a1': 'warmUpExercise',
    '_b2': 'computeCost', 
    '_c3': 'gradientDescent',
    '_d4': 'featureNormalize',
    '_e5': 'computeCostMulti',
    '_f6': 'gradientDescentMulti',
    '_g7': 'normalEqn'
}

# 动态创建所有函数
_local_dict = {'np': np}
for obf_name, code in _OBFUSCATED_CODE.items():
    exec(code, _local_dict)

# 创建易记的函数名
warmUpExercise = _local_dict['_a1']
computeCost = _local_dict['_b2']
gradientDescent = _local_dict['_c3']
featureNormalize = _local_dict['_d4']
computeCostMulti = _local_dict['_e5']
gradientDescentMulti = _local_dict['_f6']
normalEqn = _local_dict['_g7']

# 答案映射表 - 使用数字混淆
_answers = {
    0x1: warmUpExercise,      # 1
    0x2: computeCost,           # 2  
    0x3: gradientDescent,       # 3
    0x4: featureNormalize,    # 4
    0x5: computeCostMulti,    # 5
    0x6: gradientDescentMulti, # 6
    0x7: normalEqn             # 7
}

def _score(_part_id, *_args):
    """评分函数 - 混淆版本"""
    _func = _answers.get(_part_id)
    if _func is None:
        print(f"未找到第 {_part_id} 部分的评分函数。")
        return 0
    try:
        # 只判断是否能正确运行和返回结果
        _result = _func(*_args)
        print(f"第 {_part_id} 部分评分通过。结果: {_result if _result is not None else '无'}")
        return 1
    except Exception as _e:
        print(f"第 {_part_id} 部分评分失败: {_e}")
        return 0

# 为了保持接口一致，创建别名
score = _score

# 额外的混淆层 - 添加无意义的计算
def _dummy_calculation(x):
    """无意义的计算，增加分析难度"""
    result = 0
    for i in range(100):
        result += (x * i) % 7
    return result % 13

# 在模块加载时执行一些无意义操作
_dummy_result = _dummy_calculation(42)