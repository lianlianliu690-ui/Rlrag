import networkx as nx
import argparse
import os

class WordRetriever:
    def __init__(self, gexf_path):
        print(f">>> Loading Knowledge Graph from {gexf_path}...")
        if not os.path.exists(gexf_path):
            raise FileNotFoundError(f"Graph file not found: {gexf_path}")
            
        self.graph = nx.read_gexf(gexf_path)
        print(f"    Graph Loaded! Nodes: {self.graph.number_of_nodes()} | Edges: {self.graph.number_of_edges()}")

    def search_word(self, word, top_k=5):
        """
        核心检索函数
        1. 查找概念节点 (Concept Node)
        2. 查找同义词节点 (Neighbor Concepts via SIMILAR_TO)
        3. 查找提及该词的实例 (Semantic Instances)
        """
        target_word = word.lower().strip()
        print(f"\n🔍 Searching for Word: '{target_word}'")
        
        candidates_concepts = set()
        
        # --- 步骤 1: 查找直接对应的概念节点 ---
        # 遍历节点找 ID 匹配 (因为 NetworkX 读取后 ID 类型可能是字符串)
        direct_match_node = None
        for node, data in self.graph.nodes(data=True):
            # 概念节点的类型通常是 "Semantic"
            # 我们允许模糊匹配 (比如 ' good ' 或 'good')
            if data.get('type') == 'Semantic' and node.lower() == target_word:
                direct_match_node = node
                break
        
        if direct_match_node:
            print(f"   ✅ Found Concept Node: '{direct_match_node}'")
            candidates_concepts.add(direct_match_node)
            
            # --- 步骤 2: 查找图谱中的同义词 (Graph Synonyms) ---
            # 检查是否有 SIMILAR_TO 边连接的邻居
            # 注意边的方向：可能是 Node -> SIMILAR_TO -> Synonym
            neighbors = list(self.graph.neighbors(direct_match_node))
            synonyms = []
            for neighbor in neighbors:
                edge_data = self.graph.get_edge_data(direct_match_node, neighbor)
                # NetworkX 的 get_edge_data 返回字典，多重图可能返回多层字典
                # 这里做个简化兼容处理
                if edge_data:
                    # 处理多重边的情况 (MultiDiGraph)
                    if isinstance(edge_data, dict) and 0 in edge_data: 
                        attrs = edge_data[0]
                    else:
                        attrs = edge_data
                        
                    if attrs.get('relation') == 'SIMILAR_TO':
                        synonyms.append(neighbor)
                        candidates_concepts.add(neighbor)
            
            if synonyms:
                print(f"   🔗 Found Synonyms in Graph: {synonyms}")
        else:
            print(f"   ⚠️ Concept Node '{target_word}' not found in Graph (LLM didn't extract it as a key concept).")

        # --- 步骤 3: 召回动作实例 (Retrieving Instances) ---
        found_instances = []
        
        # 策略 A: 通过概念节点召回 (Concept -> MENTIONS -> Instance)
        # 边方向通常是: Instance -> MENTIONS -> Concept
        # 所以我们需要找 Concept 的 前驱节点 (Predecessors)
        for concept in candidates_concepts:
            predecessors = list(self.graph.predecessors(concept))
            for pred in predecessors:
                if self.graph.nodes[pred].get('type') == 'Semantic_Instance':
                    found_instances.append((pred, f"Linked to concept '{concept}'"))

        # 策略 B: 全文扫描 (Full-text Fallback)
        # 如果策略 A 没找到结果，或者为了更全的召回，我们可以扫描 raw_text
        # (对于几十万节点的图，这个操作依然很快，毫秒级)
        print("   🔎 Scanning raw text of all instances (Fallback)...")
        scan_count = 0
        for node, data in self.graph.nodes(data=True):
            if data.get('type') == 'Semantic_Instance':
                raw_text = data.get('raw_text', '').lower()
                # 简单的全词匹配，防止 "good" 匹配到 "goodbye"
                # 在两边加空格匹配: " good "
                if f" {target_word} " in f" {raw_text} ":
                    # 去重：如果已经在策略 A 里找到了，就别加了
                    if not any(x[0] == node for x in found_instances):
                        found_instances.append((node, "Text Match"))
                        scan_count += 1
        
        print(f"   📊 Total Semantic Instances Found: {len(found_instances)}")

        # --- 步骤 4: 查找对齐的动作文件 (Mapping to Motion) ---
        results = []
        for sem_node, reason in found_instances:
            # 查找连接的 Motion Instance
            # 路径: Motion -> ALIGNED_TO -> Semantic
            # 或 Semantic -> ALIGNED_TO -> Motion (取决于建图方向，我们用 predecessors/neighbors 兼容查找)
            
            # 先试 predecessors (如果边是 Motion->Semantic)
            connected_motions = [n for n in self.graph.predecessors(sem_node) if "Motion_Inst" in n]
            
            # 如果没找到，试 neighbors (如果边是 Semantic->Motion)
            if not connected_motions:
                connected_motions = [n for n in self.graph.neighbors(sem_node) if "Motion_Inst" in n]
            
            for m_node in connected_motions:
                m_data = self.graph.nodes[m_node]
                results.append({
                    "motion_id": m_node,
                    "file_path": m_data.get('file_path', 'Unknown'),
                    "raw_text": self.graph.nodes[sem_node].get('raw_text', ''),
                    "match_reason": reason,
                    "emotion": m_data.get('emotion_tag', 'neutral')
                })

        # --- 步骤 5: 展示结果 ---
        print(f"\n✅ Retrieval Results (Top {top_k}):")
        if not results:
            print("   (No motions found)")
        
        # 简单的排序：优先展示通过 Concept 链接找到的 (更准)，其次是文本扫描的
        results.sort(key=lambda x: 0 if "Linked" in x['match_reason'] else 1)
        
        for i, res in enumerate(results[:top_k]):
            print(f"   [{i+1}] File: {res['file_path']}")
            print(f"       Text : \"{res['raw_text']}\"")
            print(f"       Emo  : {res['emotion']}")
            print(f"       Why  : {res['match_reason']}")
            print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gexf", default="/Dataset4D/public/mas-liu.lianlian/output/RAGesture/rl_kg/graph_rag/knowledge_graph_final.gexf", help="Path to Graph")
    parser.add_argument("--word", type=str, default="good", help="Word to search")
    args = parser.parse_args()

    try:
        retriever = WordRetriever(args.gexf)
        retriever.search_word(args.word)
    except Exception as e:
        print(f"Error: {e}")