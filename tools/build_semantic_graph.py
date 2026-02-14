import os
import json
import textgrid
import pandas as pd
import networkx as nx
import argparse
from openai import OpenAI
import numpy as np

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from dotenv import load_dotenv


import torch
import sys
from mmcv import Config
# 添加必要的路径到Python路径
sys.path.insert(0, '/home/mas-liu.lianlian/RLrag')
from mogen.datasets.builder import build_dataloader, build_dataset


# 补丁: 解决 numpy float 问题
if not hasattr(np, 'float'):
    np.float = float

# 加载环境变量
load_dotenv()

# ==========================================
# 1. 动作切片索引构建器 (基于 Dataloader)
# ==========================================
# ==========================================
# 1. 动作切片索引构建器 (基于 Dataloader + 绝对时间)
# ==========================================
class MotionDataloaderIndex:
    """
    使用 Dataloader 读取 LMDB 中预存的 'abs_start_time'。
    这是最精确的对齐方式，完全消除了步长计算误差。
    """
    def __init__(self, dataset_cfg_path, device='cuda'):
        self.dataset_cfg_path = dataset_cfg_path
        self.device = device
        self.index = defaultdict(list) 
        
        print(f"Loading Config from {dataset_cfg_path}...")
        self.cfg = Config.fromfile(dataset_cfg_path)
        
        # 我们仍然需要 FPS 来计算结束时间 (Duration = Length / FPS)
        self.fps = self.cfg.get('pose_fps', 15)  # 15
        if hasattr(self.cfg.data, 'train') and hasattr(self.cfg.data.train, 'fps'):
             self.fps = self.cfg.data.train.fps
             
        print(f">>> Dataset Params | FPS: {self.fps}")
        print(">>> Note: Using pre-calculated 'abs_start_time' from LMDB.")

        self.dataloader = self._load_dataloader(dataset_cfg_path)
        self._build_index()

    def _load_dataloader(self, cfg_path):
        """加载 Dataloader (保持不变)"""
        cfg = Config.fromfile(cfg_path)
        dataset = build_dataset(cfg.data.train) # 确保读取的是包含新字段的数据集
        dataloader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=4,
            dist=False,
            shuffle=False 
        )
        return dataloader

    def _build_index(self):
        """遍历 Dataloader，直接提取绝对时间"""
        print(">>> Indexing Slices using 'abs_start_time'...")
        
        for batch in tqdm(self.dataloader, desc="Indexing"):
            try:
                # 1. 获取 ID (处理 batch 列表封装)
                s_idx = batch['sample_name']    #'2_scott_1_10_10/0'
                if isinstance(s_idx, list): s_idx = s_idx[0]
                if isinstance(s_idx, dict): s_idx = s_idx.get('id', 'unknown')
                
                slice_id = str(s_idx).replace('/', '_') # e.g., ''2_scott_1_10_10_0''
                
                # 2. 推导 Parent ID
                # 依然需要这个来把切片归类到同一个长录音下
                parts = slice_id.split('_')
                if parts[-1].isdigit():
                    parent_id = "_".join(parts[:-1]) # "'2_scott_1_10_10'"
                else:
                    parent_id = slice_id

                # 3. 🔥 [核心修改] 直接获取绝对开始时间
                # 如果您重新生成了数据，batch 中一定有 'abs_start_time'
                if 'abs_start_time' in batch:
                    abs_start = batch['abs_start_time']
                    
                    # Dataloader 可能会把 float 变成 Tensor，这里做个转换
                    if isinstance(abs_start, torch.Tensor):
                        abs_start = abs_start.item()
                    elif isinstance(abs_start, list):
                        abs_start = float(abs_start[0])
                    else:
                        abs_start = float(abs_start)
                
                # 4. 计算结束时间
                # End = Start + (FrameLen / FPS)
                current_len = 150 # 默认值
                if 'motion' in batch:
                    # 获取由 dataset 返回的真实 motion 长度
                    # shape 通常是 [batch, frames, dim]
                    if hasattr(batch['motion'], 'shape'):
                         # 启发式判断哪个是时间维度 (通常 > 10)
                        shape = batch['motion'].shape
                        current_len = shape[1]
                        
                
                duration = current_len / self.fps
                abs_end = abs_start + duration

                # 5. 存入索引
                self.index[parent_id].append({
                    'slice_id': slice_id,   #2_scott_1_10_10_0
                    'start': float(abs_start), # 绝对时间
                    'end': float(abs_end)      # 绝对时间
                })
                
            except Exception as e:
                # print(f"Error indexing batch {s_idx}: {e}")
                continue
        
        print(f"Index built. Found {len(self.index)} parent recordings.")

    def get_slices(self, parent_id):
        return self.index.get(parent_id, [])

# ==========================================
# 2. TextGrid 读取器 (完整上下文) - 保持不变
# ==========================================
class FilteredTextGridReader:
    """
    功能 1: 根据 CSV 筛选特定 Split (train/test) 和 Speaker 的文件 ID。
    功能 2: 读取这些文件的完整上下文和单词时间戳。
    """
    def __init__(self, csv_path, textgrid_dir, target_speakers=[2], split_type='train'):
        self.textgrid_dir = textgrid_dir
        
        # --- 1. 恢复筛选逻辑 ---
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")
        
        print(f"Loading split rules from: {csv_path}")
        try:
            split_rule = pd.read_csv(csv_path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV: {e}")

        # 提取 Speaker ID (假设格式 "2_scott_...")
        split_rule['speaker_id_int'] = split_rule['id'].astype(str).str.split("_").str[0].astype(int)

        # 执行筛选: Type == split_type AND SpeakerID in target_speakers
        self.selected_df = split_rule.loc[
            (split_rule['type'] == split_type) & 
            (split_rule['speaker_id_int'].isin(target_speakers))
        ]
        
        # 获取合法的文件 ID 列表
        # 注意：CSV 里的 ID 通常是 "2_scott_0_1_0" (切片ID) 还是 "2_scott_0_1" (长音频ID)?
        # 如果 CSV 里存的是切片 ID，我们需要去重得到父 ID
        raw_ids = self.selected_df['id'].tolist()
        
        # 预处理：提取父文件名 (Parent ID)
        # 假设 csv 里是 2_scott_0_1_0，我们要提取 2_scott_0_1
        # 如果 csv 里本身就是长音频 ID，这一步也不会出错
        unique_parent_ids = set()
        for rid in raw_ids:
            parts = str(rid).split('_')
            # 简单的启发式规则：通常最后一位是切片索引
            # 如果文件名像 2_scott_0_1，则它本身就是父ID
            # 如果文件名像 2_scott_0_1_0，则父ID是 2_scott_0_1
            # 这里我们为了保险，检查对应 TextGrid 是否存在
            
            # 尝试直接用 ID
            if os.path.exists(os.path.join(textgrid_dir, f"{rid}.TextGrid")):
                unique_parent_ids.add(rid)
            else:
                # 尝试去掉最后一位作为父 ID
                parent_candidate = "_".join(parts[:-1])
                if os.path.exists(os.path.join(textgrid_dir, f"{parent_candidate}.TextGrid")):
                    unique_parent_ids.add(parent_candidate)
        
        self.file_ids = list(unique_parent_ids)
        print(f"Filter Result: Found {len(self.file_ids)} unique Parent TextGrids (Split={split_type}).")

    def get_files(self):
        return self.file_ids

    def read_full_text(self, file_id):
        """
        读取完整上下文 (保留原 FullContextTextGridReader 的逻辑)
        """
        path = os.path.join(self.textgrid_dir, f"{file_id}.TextGrid")
        if not os.path.exists(path):
            return None, None
            
        try:
            tg = textgrid.TextGrid.fromFile(path)
            full_text_list = []
            words = [] 
            
            target_tiers = ['words', 'transcript', 'word', 'upper_word']
            
            for tier in tg:
                if tier.name.lower() in target_tiers:
                    for interval in tier:
                        w = interval.mark.strip()
                        if w and w.lower() not in ['<sil>', 'sil', 'sp', '']:
                            full_text_list.append(w)
                            words.append({
                                'word': w,
                                'start': interval.minTime,
                                'end': interval.maxTime
                            })
                    break 
            
            return " ".join(full_text_list), words
        except Exception as e:
            # print(f"Error reading TextGrid {file_id}: {e}")
            return None, None

# ==========================================
# 2. 语义提取模块 (保持不变)
# ==========================================
class SemanticExtractor:
    def __init__(self, model="qwen-plus"):
        
        self.client = OpenAI(
            # 使用 DashScope API Key
            api_key="sk-22f3c8747a014e7d81c1678a1d39817e", 
            # 关键点：将 base_url 指向阿里云的兼容服务端点
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = model

    def extract_triplets(self, text):
        prompt = f"""
        You are an expert Linguistic Analyst building the Semantic Layer of a Knowledge Graph for a 3D digital human system.
        Your goal is to parse the input speech text into a structured graph that captures **Content**, **Logic**, and **Emotion**.

        Input Text: "{text}"

        ### 1. Schema Definition (Strictly Follow)

        **Node Types**:
        1. "Semantic": Concrete words found in the text (Verbs, Nouns, Adjectives). Normalize lemmas (e.g., "running" -> "run").
        2. "Discourse_Function": Abstract logic nodes. **ONLY use these values**: 
        ["CONTRAST", "CAUSAL", "EMPHASIS", "UNCERTAINTY", "AGREEMENT", "ELABORATION"].
        3. "Emotion": Abstract emotion nodes. **ONLY use these values**: 
        ["happiness", "surprise", "sadness", "neutral", "anger", "contempt", "fear", "disgust"].

        **Edge Relations**:
        1. "BELONGS_TO": Connects a structure word (e.g., "but") to a "Discourse_Function" (e.g., "CONTRAST").
        2. "EXPRESSES": Connects a content word (e.g., "furious") to an "Emotion" (e.g., "anger").
        3. "SIMILAR_TO": Generalization. Connects a specific word to a more common synonym (e.g., "furious" -> "angry").
        4. "IS_A": Hierarchy. Connects a specific concept to a general category (e.g., "apple" -> "fruit").
        5. "CAUSES": [因果/上下文] Contextual link between two semantic events. (e.g., "accident" -> "shock").

        ### 2. Tasks

        1. **Extract Nodes**: Identify all key content words and logic words. Create "Semantic" nodes for them.
        2. **Map Logic**: If a word indicates a shift in logic (e.g., "however", "so", "actually"), create a "Discourse_Function" node and link them.
        3. **Map Emotion**: If a word carries strong sentiment, create an "Emotion" node and link them.
        4. **Expand Knowledge**: For key content words, add "SIMILAR_TO" or "IS_A" edges to general concepts. This helps the system find gestures even for rare words.

        ### 3. Output Format
        Output **strictly valid JSON** containing "nodes" and "edges".

        #### Example Input:
        "I was absolutely furious, but I stayed silent."

        #### Example Output:
        {{
            "nodes": [
                {{"id": "furious", "type": "Semantic"}},
                {{"id": "absolutely", "type": "Semantic"}},
                {{"id": "but", "type": "Semantic"}},
                {{"id": "silent", "type": "Semantic"}},
                {{"id": "anger", "type": "Emotion"}},
                {{"id": "CONTRAST", "type": "Discourse_Function"}},
                {{"id": "EMPHASIS", "type": "Discourse_Function"}}
            ],
            "edges": [
                {{"source": "furious", "target": "anger", "relation": "EXPRESSES", "weight": 1.0}},
                {{"source": "furious", "target": "angry", "relation": "SIMILAR_TO", "weight": 0.9}},
                {{"source": "but", "target": "CONTRAST", "relation": "BELONGS_TO", "weight": 1.0}},
                {{"source": "absolutely", "target": "EMPHASIS", "relation": "BELONGS_TO", "weight": 0.8}},
                {{"source": "silent", "target": "quiet", "relation": "SIMILAR_TO", "weight": 0.7}}
            ]
        }}
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    # Qwen 有时需要 System Prompt 来增强指令遵循
                    {"role": "system", "content": "You are a helpful assistant specialized in knowledge graph extraction. Output strictly in JSON format."}, 
                    {"role": "user", "content": prompt}
                ],
                # Qwen 的兼容接口支持 response_format={"type": "json_object"}
                # 但如果遇到报错，可以尝试去掉这行，Qwen 对 Prompt 的遵循能力通常足够
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            # 解析返回内容
            content = response.choices[0].message.content
            # 有时候模型可能返回 ```json ... ``` 包裹的格式，做一个简单的清洗更稳健
            if content.startswith("```json"):
                content = content.strip("```json").strip("```")
            elif content.startswith("```"):
                content = content.strip("```")
                
            return json.loads(content)
            
        except Exception as e:
            print(f"Qwen API Error: {e}")
            return None 

# ==========================================
# 3. 图谱构建模块 (保持不变)
# ==========================================
class SemanticGraphBuilder:
    def __init__(self, output_dir):
        self.graph = nx.DiGraph()
        self.output_dir = output_dir
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        self._init_fixed_nodes()

    def _init_fixed_nodes(self):
        emotions = ["happiness", "sadness", "neutral", "anger", "fear", "surprise", "disgust", "contempt"]
        for e in emotions:
            self.graph.add_node(e, type="Emotion", is_fixed=True)

    def update(self, graph_data, parent_id, slices_info, word_timings):
        """
        graph_data: LLM 全局分析结果
        parent_id: "2_scott_0_1"
        slices_info: 来自 Dataloader 的切片列表
        word_timings: 来自 TextGrid 的单词时间表
        """
        if not graph_data: return

        # 1. 添加全局概念节点
        global_concepts = set()
        for node in graph_data.get("nodes", []):
            nid = str(node.get("id", "")).lower().strip()
            if nid:
                global_concepts.add(nid)
                if not self.graph.has_node(nid):
                    self.graph.add_node(nid, type=node.get("type", "Semantic"))
        
        for edge in graph_data.get("edges", []):
            s = str(edge.get("source")).lower().strip()
            t = str(edge.get("target")).lower().strip()
            if s and t: self.graph.add_edge(s, t, relation=edge.get("relation"))

        processed_concepts = {}
        for c in global_concepts:
            # 将 "playing_game" 处理为 "playing game"
            # 将 "pick_up" 处理为 "pick up"
            clean_c = c.replace("_", " ").lower().strip()
            processed_concepts[clean_c] = c # Value 必须是原始 ID (带下划线的)

        # 2. 构建单词时间查找表 (加速匹配)
        # word -> [(start, end), ...]
        word_map = defaultdict(list)
        for item in word_timings:
            w_clean = str(item['word']).lower().strip()
            word_map[w_clean].append((float(item['start']), float(item['end'])))

        # 3. 为每个 Motion Slice 创建 Semantic Instance
        for sl in slices_info:
            slice_id = sl['slice_id'] # e.g., 2_scott_0_1_0
            s_start = sl['start']
            s_end = sl['end']
            
            inst_id = f"Semantic_Inst_{slice_id}"   #'Semantic_Inst_2_scott_0_51_51_0'
            
            # --- 步骤 A: 收集落在该切片内的所有单词 ---
            slice_word_objs = [] 
            
            for word, timings_list in word_map.items():
                for (w_start, w_end) in timings_list:
                    # 判定重叠: 不相离即重叠
                    if not (w_end < s_start or w_start > s_end):
                        slice_word_objs.append({
                            'word': word,
                            'start': w_start
                        })
            
            # --- 步骤 B: 按时间排序并重组句子 ---
            # 按时间排序，保证语序正确 (e.g. "pick" then "up")
            slice_word_objs.sort(key=lambda x: x['start'])
            
            # 提取纯文本列表
            slice_words_list = [obj['word'] for obj in slice_word_objs]
            
            # 拼成完整字符串，前后加空格方便全词匹配
            # e.g. " i am playing game "
            slice_full_text = " " + " ".join(slice_words_list).lower() + " "
            
            # --- 步骤 C: 短语级匹配 (Phrase Matching) ---
            related_concepts = set()
            
            # 遍历所有全局概念，看它们是否出现在这一小段文本里
            for clean_concept, original_concept_id in processed_concepts.items():
                # 检查 " playing game " 是否在 " i am playing game " 里
                # 这里做简单的子串匹配，对于大多数短语足够了
                if clean_concept in slice_full_text:
                    related_concepts.add(original_concept_id)
                
                # 特殊情况：如果是单个词，防止匹配到单词的一部分
                # 例如 concept="act"，防止匹配到 "actually"
                # 可以加空格判断: " act " in slice_full_text
                elif " " + clean_concept + " " in slice_full_text:
                     related_concepts.add(original_concept_id)
            
            # 创建实例节点
            self.graph.add_node(
                inst_id,
                type="Semantic_Instance",
                base_id=slice_id,
                parent_id=parent_id,
                start_time=s_start,
                end_time=s_end,
                raw_text=" ".join(slice_words_list)
            )
            
            # 连接 Instance -> Global Concepts
            for concept in related_concepts:
                self.graph.add_edge(inst_id, concept, relation="MENTIONS")
            
            # 连接 Instance -> Global Emotions (Context)
            for node in graph_data.get("nodes", []):
                if node.get("type") == "Emotion":
                    e_nid = str(node.get("id")).lower().strip()
                    self.graph.add_edge(inst_id, e_nid, relation="HAS_CONTEXT_EMOTION")

    def save(self):
        nx.write_gexf(self.graph, os.path.join(self.output_dir, "semantic_layer.gexf"))

# ==========================================
# 5. 主流程
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_cfg", required=True, help="Path to dataset config")
    
    # 恢复 CSV 参数
    parser.add_argument("--csv_path", required=True, help="Path to train_test_split.csv")
    parser.add_argument("--split", default="train", help="train/test/val")
    parser.add_argument("--speaker_id", type=int, default=2)
    
    parser.add_argument("--textgrid_dir", required=True, help="Path to TextGrid directory")
    parser.add_argument("--output_dir", default="data/graph_rag/semantic_final")
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    # 1. 建立动作索引 (从 Dataloader 获取切片信息)
    print("--- 1. Building Motion Index from Dataloader ---")
    motion_index = MotionDataloaderIndex(args.dataset_cfg)
    
    # 2. 读取 TextGrid (带筛选功能)
    print(f"--- 2. Reading TextGrids (Split: {args.split}) ---")
    # 使用合并后的 FilteredTextGridReader
    tg_reader = FilteredTextGridReader(
        csv_path=args.csv_path,
        textgrid_dir=args.textgrid_dir,
        target_speakers=[args.speaker_id],
        split_type=args.split
    )
    files = tg_reader.get_files()
    
    print(f"Found {len(files)} TextGrid files for semantic analysis.")

    # 3. 并行处理
    extractor = SemanticExtractor()
    builder = SemanticGraphBuilder(args.output_dir)
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {}
        
        for pid in files:
            full_text, word_timings = tg_reader.read_full_text(pid)
            if not full_text: continue
            
            # 提交 LLM 任务
            future = executor.submit(extractor.extract_triplets, full_text)
            future_map[future] = (pid, word_timings)
            
        for future in tqdm(as_completed(future_map), total=len(future_map), desc="Building Graph"):
            parent_id, word_timings = future_map[future]
            graph_data = future.result()
            
            # 从索引中获取切片信息
            slices_info = motion_index.get_slices(parent_id)
            
            if not slices_info:
                # 可能是因为 dataset split 过滤掉了某些文件
                continue
                
            builder.update(graph_data, parent_id, slices_info, word_timings)

    builder.save()
    print("Done!")

if __name__ == "__main__":
    import debugpy
    try:
        # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
        debugpy.listen(("localhost", 9502))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
    except Exception as e:
      pass
    main()