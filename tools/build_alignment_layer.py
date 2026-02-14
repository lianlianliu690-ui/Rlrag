import os
import sys
import networkx as nx
import argparse
from tqdm import tqdm
import numpy as np

# --- 补丁: 解决 numpy float 兼容性问题 ---
if not hasattr(np, 'float'):
    np.float = float
# --- 补丁结束 ---

class TimeAwareGraphAligner:
    def __init__(self, semantic_path, motion_path, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f">>> 1. Loading Graphs...")
        
        # 1. 加载语义层
        print(f"    Reading Semantic Graph: {semantic_path}")
        try:
            self.semantic_graph = nx.read_gexf(semantic_path)
            print(f"    ✅ Nodes: {self.semantic_graph.number_of_nodes()}")
        except Exception as e:
            print(f"❌ Failed to load semantic graph: {e}")
            sys.exit(1)

        # 2. 加载动作层
        print(f"    Reading Motion Graph: {motion_path}")
        try:
            self.motion_graph = nx.read_gexf(motion_path)
            print(f"    ✅ Nodes: {self.motion_graph.number_of_nodes()}")
        except Exception as e:
            print(f"❌ Failed to load motion graph: {e}")
            sys.exit(1)

        # 3. 合并图谱
        print(">>> 2. Merging Graphs (Compose)...")
        self.merged_graph = nx.compose(self.semantic_graph, self.motion_graph)

    def calculate_overlap(self, start1, end1, start2, end2):
        """计算两个时间段的重叠秒数"""
        if end1 <= start2 or end2 <= start1:
            return 0.0
        
        intersection_start = max(start1, start2)
        intersection_end = min(end1, end2)
        return max(0.0, intersection_end - intersection_start)

    def align(self, tolerance=0.1):
        """
        核心对齐逻辑 + 自动清洗:
        1. 对齐: Motion <-> Semantic
        2. 清洗: 删除所有未能对齐的 Motion 节点
        """
        print(">>> 3. Aligning Instances (ID Match + Time Verify)...")
        
        aligned_count = 0
        time_mismatch_count = 0
        
        # 🔥 用于记录成功对齐的动作节点 ID
        matched_motion_ids = set()
        
        all_nodes = list(self.merged_graph.nodes(data=True))
        
        for node_id, data in tqdm(all_nodes, desc="Linking"):
            
            # 1. 筛选动作节点
            if not (node_id.startswith("Motion_Inst_") or data.get('type') == 'Motion_Instance'):
                continue

            # 2. 提取 Core ID
            core_id = node_id.replace("Motion_Inst_", "")
            
            # 3. 构造目标 Semantic ID
            target_semantic_id = f"Semantic_Inst_{core_id}"
            
            # 4. 检查与校验
            if self.merged_graph.has_node(target_semantic_id):
                semantic_data = self.merged_graph.nodes[target_semantic_id]
                
                try:
                    m_start = float(data.get('start_time', -1))
                    m_end = float(data.get('end_time', -1))
                    s_start = float(semantic_data.get('start_time', -2))
                    s_end = float(semantic_data.get('end_time', -2))
                    
                    time_aligned = False
                    
                    # 校验逻辑
                    if abs(m_start - s_start) < tolerance:
                        time_aligned = True
                    else:
                        overlap = self.calculate_overlap(m_start, m_end, s_start, s_end)
                        duration = m_end - m_start
                        if duration > 0 and (overlap / duration) > 0.8:
                            time_aligned = True
                    
                    if time_aligned:
                        self.merged_graph.add_edge(
                            node_id, 
                            target_semantic_id, 
                            relation="ALIGNED_TO", 
                            type="alignment_edge",
                            weight=1.0
                        )
                        aligned_count += 1
                        matched_motion_ids.add(node_id) # 🔥 记录成功匹配的 ID
                    else:
                        time_mismatch_count += 1
                        
                except Exception:
                    pass
            else:
                pass # 语义层不存在，说明是静音片段

        # ==========================================
        # 🔥 新增: 清洗未匹配节点 (Pruning)
        # ==========================================
        print(f"\n>>> 4. Cleaning up unmatched motions (Silence/Noise)...")
        
        # 再次遍历所有节点，找出所有是 Motion 但不在 matched_motion_ids 里的
        nodes_to_remove = []
        # 注意：这里需要重新获取所有节点，或者复用之前的逻辑
        # 为了安全，我们只删除 "Motion_Instance" 类型的未匹配节点
        
        current_nodes = list(self.merged_graph.nodes(data=True))
        for node_id, data in current_nodes:
            # 判定它是动作节点
            is_motion = node_id.startswith("Motion_Inst_") or data.get('type') == 'Motion_Instance'
            
            if is_motion:
                if node_id not in matched_motion_ids:
                    nodes_to_remove.append(node_id)
        
        # 执行批量删除
        self.merged_graph.remove_nodes_from(nodes_to_remove)
        
        print(f"✅ Alignment & Cleaning Complete!")
        print(f"   -----------------------------------------")
        print(f"   Matched Pairs (Kept)    : {aligned_count}")
        print(f"   Unmatched Removed       : {len(nodes_to_remove)} (Deleted)")
        print(f"   Time Mismatches         : {time_mismatch_count}")
        print(f"   -----------------------------------------")
        print(f"   Final Graph Nodes       : {self.merged_graph.number_of_nodes()}")
        print(f"   Final Graph Edges       : {self.merged_graph.number_of_edges()}")

    def save(self):
        output_path = os.path.join(self.output_dir, "knowledge_graph_final_clean.gexf")
        print(f">>> 5. Saving Final Knowledge Graph to {output_path}...")
        nx.write_gexf(self.merged_graph, output_path)
        print(f"🎉 Graph saved! You can visualize it in Gephi.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic_gexf", required=True, help="Path to semantic_layer.gexf")
    parser.add_argument("--motion_gexf", required=True, help="Path to motion_instance_layer.gexf")
    parser.add_argument("--output_dir", default="data/graph_rag/final_kg", help="Directory to save the final graph")

    args = parser.parse_args()

    aligner = TimeAwareGraphAligner(
        semantic_path=args.semantic_gexf,
        motion_path=args.motion_gexf,
        output_dir=args.output_dir
    )
    
    aligner.align()
    aligner.save()