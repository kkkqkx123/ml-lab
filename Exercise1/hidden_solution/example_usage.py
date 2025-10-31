"""
示例用法 - 展示如何使用隐藏答案模块
"""
import numpy as np
import sys
import os

# 将隐藏解决方案目录添加到路径
sys.path.insert(0, os.path.dirname(__file__))

def test_encoded_solution():
    """测试编码解决方案"""
    print("=== 测试编码解决方案 ===")
    try:
        from encoded_answers import score, warmUpExercise, computeCost, gradientDescent
        
        # 测试 warmUpExercise
        print("测试 warmUpExercise:")
        result = warmUpExercise()
        print(f"结果形状: {result.shape}")
        print(f"结果类型: {type(result)}")
        
        # 测试评分函数
        print("\n测试评分函数:")
        score_result = score(1)
        print(f"评分结果: {score_result}")
        
        # 测试 computeCost
        print("\n测试 computeCost:")
        X = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
        y = np.array([2, 4, 6, 8])
        theta = np.array([0, 1])
        cost = computeCost(X, y, theta)
        print(f"计算成本: {cost}")
        
    except Exception as e:
        print(f"编码解决方案测试失败: {e}")

def test_obfuscated_solution():
    """测试混淆解决方案"""
    print("\n=== 测试混淆解决方案 ===")
    try:
        from obfuscated_answers import score, warmUpExercise, computeCost
        
        # 测试 warmUpExercise
        print("测试 warmUpExercise:")
        result = warmUpExercise()
        print(f"结果形状: {result.shape}")
        print(f"结果类型: {type(result)}")
        
        # 测试评分函数
        print("\n测试评分函数:")
        score_result = score(1)
        print(f"评分结果: {score_result}")
        
    except Exception as e:
        print(f"混淆解决方案测试失败: {e}")

def test_encrypted_solution():
    """测试加密解决方案"""
    print("\n=== 测试加密解决方案 ===")
    try:
        from encrypted_answers import score, warmUpExercise, computeCost
        
        # 测试 warmUpExercise
        print("测试 warmUpExercise:")
        result = warmUpExercise()
        print(f"结果形状: {result.shape}")
        print(f"结果类型: {type(result)}")
        
        # 测试评分函数
        print("\n测试评分函数:")
        score_result = score(1)
        print(f"评分结果: {score_result}")
        
    except Exception as e:
        print(f"加密解决方案测试失败: {e}")

def compare_with_original():
    """与原始答案比较"""
    print("\n=== 与原始答案比较 ===")
    try:
        # 导入原始答案
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from answers import warmUpExercise as original_warmUpExercise
        from encoded_answers import warmUpExercise as encoded_warmUpExercise
        
        # 比较结果
        original_result = original_warmUpExercise()
        encoded_result = encoded_warmUpExercise()
        
        print(f"原始结果:\n{original_result}")
        print(f"编码结果:\n{encoded_result}")
        print(f"结果是否相同: {np.array_equal(original_result, encoded_result)}")
        
    except Exception as e:
        print(f"比较失败: {e}")

def show_code_visibility():
    """展示代码可见性"""
    print("\n=== 代码可见性分析 ===")
    
    # 检查原始文件
    original_file = os.path.join(os.path.dirname(__file__), '..', 'answers.py')
    if os.path.exists(original_file):
        with open(original_file, 'r', encoding='utf-8') as f:
            original_content = f.read()
        print(f"原始文件大小: {len(original_content)} 字符")
        print("原始文件可以直接看到函数实现")
    
    # 检查编码文件
    encoded_file = os.path.join(os.path.dirname(__file__), 'encoded_answers.py')
    if os.path.exists(encoded_file):
        with open(encoded_file, 'r', encoding='utf-8') as f:
            encoded_content = f.read()
        print(f"编码文件大小: {len(encoded_content)} 字符")
        print("编码文件使用动态执行，函数实现不可直接见")
    
    # 检查混淆文件
    obfuscated_file = os.path.join(os.path.dirname(__file__), 'obfuscated_answers.py')
    if os.path.exists(obfuscated_file):
        with open(obfuscated_file, 'r', encoding='utf-8') as f:
            obfuscated_content = f.read()
        print(f"混淆文件大小: {len(obfuscated_content)} 字符")
        print("混淆文件使用变量名混淆，增加阅读难度")
    
    # 检查加密文件
    encrypted_file = os.path.join(os.path.dirname(__file__), 'encrypted_answers.py')
    if os.path.exists(encrypted_file):
        with open(encrypted_file, 'r', encoding='utf-8') as f:
            encrypted_content = f.read()
        print(f"加密文件大小: {len(encrypted_content)} 字符")
        print("加密文件使用base64编码函数字符串")

if __name__ == "__main__":
    print("隐藏答案解决方案测试")
    print("=" * 50)
    
    test_encoded_solution()
    test_obfuscated_solution()
    test_encrypted_solution()
    compare_with_original()
    show_code_visibility()
    
    print("\n" + "=" * 50)
    print("测试完成！")
    print("\n使用建议:")
    print("1. encoded_answers.py - 推荐，平衡了安全性和性能")
    print("2. obfuscated_answers.py - 中等安全性，增加阅读难度")
    print("3. encrypted_answers.py - 较高安全性，但性能略低")
    print("\n所有方案都保持了与原始代码相同的功能！")