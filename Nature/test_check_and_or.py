from generate_net import graph_generation
import numpy as np

def test_check_and_or():
    """测试check_and_or函数"""
    print("测试check_and_or函数...")
    
    # 创建一个小型网络进行测试
    np.random.seed(12345)
    edges, end_point, length, net = graph_generation(5)
    
    print(f"生成的网络有 {len(net.points)} 个节点")
    print(f"网络边: {edges}")
    print(f"AND/OR设置: {net.andor}")
    
    # 打印每个节点的信息
    for i, point in enumerate(net.points):
        children_locs = [c.loc for c in point.children]
        node_type = getattr(point, 'type', 'None')
        print(f"节点 {i}: type={node_type}, children={children_locs}")
    
    # 测试check_and_or函数
    print("\n执行check_and_or验证...")
    result = net.check_and_or()
    print(f"验证结果: {result}")
    
    print(f"修正后的AND/OR设置: {net.andor}")
    
    # 再次打印每个节点的信息
    print("\n修正后的节点信息:")
    for i, point in enumerate(net.points):
        children_locs = [c.loc for c in point.children]
        node_type = getattr(point, 'type', 'None')
        print(f"节点 {i}: type={node_type}, children={children_locs}")

if __name__ == "__main__":
    test_check_and_or()
