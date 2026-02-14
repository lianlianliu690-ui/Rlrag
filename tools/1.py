import networkx as nx
import os
import sys

# 设置文件路径
GEXF_PATH = "/Dataset4D/public/mas-liu.lianlian/output/RAGesture/rl_kg/graph_rag/rag_graph_final.gexf"

def analyze_graph_coverage(path):
    if not os.path.exists(path):
        print(f"[Error] 文件不存在: {path}")
        return

    print(f">>> 正在加载图谱: {path} ...")
    try:
        # 读取 GEXF 文件
        g = nx.read_gexf(path)
    except Exception as e:
        print(f"[Error] 加载失败: {e}")
        return

    print(f"    - 节点总数: {len(g.nodes())}")
    print(f"    - 边总数: {len(g.edges())}")
    print(f"    - 图类型: {'有向图 (DiGraph)' if g.is_directed() else '无向图 (Graph)'}")

    # 1. 区分动作节点和语义节点
    motion_nodes = []
    semantic_nodes = []
    
    # 根据之前的命名规则 "Motion_Primitive_XXX" 来识别
    for node in g.nodes():
        node_str = str(node)
        if node_str.startswith("Motion_Primitive_"):
            motion_nodes.append(node_str)
        else:
            semantic_nodes.append(node_str)

    num_motions = len(motion_nodes)
    print(f"\n>>> 节点分类统计:")
    print(f"    - 动作节点 (Motion Primitives): {num_motions}")
    print(f"    - 语义节点 (Semantic Words): {len(semantic_nodes)}")

    if num_motions == 0:
        print("[Warning] 未找到动作节点，请检查节点命名规则是否变更。")
        return

    # 2. 统计连接覆盖率
    # 之前的代码逻辑是: add_edge(word, motion_id) -> 方向是 Word -> Motion
    # 所以我们要检查动作节点的“入度邻居” (Predecessors)
    
    covered_motions = []       # 有对应词的动作
    uncovered_motions = []     # 没有对应词的动作 (孤立动作)
    
    # 缓存所有动作节点的集合，用于快速判断邻居是否为动作（排除动作间的边）
    motion_node_set = set(motion_nodes)

    for m_node in motion_nodes:
        connected_words = set()
        
        # 获取所有连接的邻居（兼容有向图和无向图）
        neighbors = []
        if g.is_directed():
            # Word 指向 Motion，所以我们要找 Predecessors (上游节点)
            neighbors.extend(list(g.predecessors(m_node)))
            # 也可以检查 Successors，以防边反了
            neighbors.extend(list(g.successors(m_node)))
        else:
            neighbors.extend(list(g.neighbors(m_node)))
            
        # 过滤：只保留连接到“非动作节点”的邻居
        for n in neighbors:
            if n not in motion_node_set:
                connected_words.add(n)
        
        if len(connected_words) > 0:
            covered_motions.append((m_node, len(connected_words), list(connected_words)[:5]))
        else:
            uncovered_motions.append(m_node)

    # 3. 输出结果
    num_covered = len(covered_motions)
    coverage_rate = (num_covered / num_motions) * 100

    print(f"\n>>> 覆盖率分析结果:")
    print(f"    --------------------------------------------------")
    print(f"    总动作数            : {num_motions}")
    print(f"    有对应语义词的动作数 : {num_covered}")
    print(f"    无对应语义词的动作数 : {len(uncovered_motions)}")
    print(f"    --------------------------------------------------")
    print(f"    ★ 动作语义覆盖率    : {coverage_rate:.2f}%")
    print(f"    --------------------------------------------------")

    # 4. 打印详细样本
    if num_covered > 0:
        print(f"\n>>> [样本] 连接数最多的 Top 5 动作:")
        # 按连接词的数量降序排列
        covered_motions.sort(key=lambda x: x[1], reverse=True)
        for m_id, count, words in covered_motions[:5]:
            print(f"    - {m_id} (关联了 {count} 个词): {words} ...")

    if len(uncovered_motions) > 0:
        print(f"\n>>> [样本] 没有任何语义关联的动作 (Top 5):")
        print(f"    {uncovered_motions[:5]}")

if __name__ == "__main__":
    analyze_graph_coverage(GEXF_PATH)