import torch
import numpy as np
import json
import re
import copy
import time
import random
import os
from openai import OpenAI
from dotenv import load_dotenv
from .utils import sort_sidx_by_textsimilarity, get_word_similarity_score, map_conns_to_prominence
from torch.nn import functional as F

# 加载环境变量
load_dotenv()

# ================= 1. 定义新的 System Prompt (双字段对齐版) =================
INTENT_REASONING_PROMPT = """
### Role
You are a Motion Director Agent. Your goal is to analyze user speech and construct a **Dynamic Intent Graph** to guide 3D character animation.

### Reasoning Process
1. **Semantic Node Extraction**: Identify words with **Strong Semantic Relevance** to physical expression.
   - **Scope**: NOT just Actions and Emotions. You must also include:
     - **Metaphors**: Words that imply a physical state (e.g., "heartbroken" -> implies clutching chest/slumping).
     - **Semantic Triggers**: Nouns/Events that trigger a reaction (e.g., "fire", "explosion" -> implies ducking/shielding face).
     - **Intense Adjectives**: Words describing extreme states (e.g., "petrified", "fuming").
   - **Context**: Identify objects/places merely as context (do not expand these).

2. **Contextual Logic (Edges)**: Analyze the relationships between nodes to refine the visual description.
   - **NEXT**: Sequential order.
   - **WHILE**: Simultaneous actions (e.g., "speaking" WHILE "walking"). -> *Visual Desc should combine both traits.*
   - **BUT**: Contrasting/Conflicting states. -> *Crucial*: Modify the visual description of the second node to show conflict (e.g., "Sad" BUT "Smile" -> Visual: "Forced, bitter smile, mouth corners up but eyes sad").
   - **CAUSED_BY**: Reactionary logic. (e.g., "Hit" CAUSED_BY "Punch").

3. **Expansion & Description**:
    - **Synonyms**: Must be **Simple, Basic English Verbs/Adjectives** likely found in standard motion datasets (e.g., use "hit", "punch", "beat" instead of "pummel", "strike").
   - **Visual Description**: Must be **Physics-Aware**. Describe specific body parts (arms, head, torso) and movement dynamics (fast, slow, shaking). Avoid abstract metaphors here.

### Output Format (JSON)
{
  "target_emotion": "String", // Must be one of ("happiness", "surprise", "sadness", "neutral", "anger", "contempt", "fear", "disgust")
  "intent_nodes": [
    {
      "id": "n0",
      // 1. 语义标签：用于搜索动作库 (可以是归纳后的词)
      "semantic_label": "sadness", 
      
      // 2. 原文锚点：必须完全复制原文中触发该意图的单词 (用于时间对齐)
      "source_anchor": "heartbroken", 
      
      "type": "metaphor", // action, emotion, metaphor, trigger
      "synonyms": ["sad", "cry", "pain"], 
      "visual_description": "clutching chest with both hands, looking down, shoulders slumped" // Physics-aware description
    },
    {
      "id": "n1",
      "semantic_label": "smile",
      "source_anchor": "smile",
      "type": "action",
      "synonyms": ["laugh", "grin"],
      "visual_description": "awkward forced smile, stiff posture, head slightly lowered" // Modified by 'BUT' logic
    }
  ],
  "edges": [
    {"source": "n0", "target": "n1", "relation": "BUT"}
  ]
}
"""

# ================= 2. 初始化 Qwen 客户端 =================
# 从环境变量获取 Key 和 Base URL
openai_api_key = os.getenv("Qwen_API_KEY") 
# 默认使用阿里云 DashScope 的兼容接口，也可以在 .env 中自定义
openai_base_url = os.getenv("Qwen_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# 初始化客户端
try:
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_base_url
    )
except Exception as e:
    print(f"[Warning] Failed to init OpenAI/Qwen client: {e}")
    client = None

# 设置默认模型名称
model_name = os.getenv("model_name", "qwen-plus") 

def call_llm_intent(text):
    """调用 LLM 进行意图推理，返回 JSON 对象"""
    if client is None:
        return None
        
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": INTENT_REASONING_PROMPT},
                {"role": "user", "content": f"Analyze the following speech: \"{text}\""}
            ],
            response_format={"type": "json_object"} # 强制 JSON 输出
        )
        content = completion.choices[0].message.content
        # 清理 Markdown 标记
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception as e:
        print(f"[LLM Error]: {e}")
        return None
def llm_retrieval(
    text,
    text_times,
    speaker_id,
    prominence,
    db_idx_2_gesture_labels,    # 旧索引，保留用于 vector fallback
    db_idx_2_prominence,
    encoded_text,
    text_feat_cache,
    kg_graph=None,              # [新增] NetworkX 图对象
    text_model=None,        # [新增] 必须传入文本编码模型 (如 MotionCLIP/TMR 的 text_encoder)
    device='cpu'            # [新增] 设备
):
    """
    混合检索主入口：
    1. 动态图构建 (Intent Reasoning)
    2. Path A: 静态图谱遍历 (Graph Traversal: Concept -> Semantic -> Motion)
    3. Path B: 向量检索 (Vector Fallback)
    """
    d_bounds = {}
    sample_indexes = {}
    query_gest_bounds = {}

    if text.strip() == "":
        return sample_indexes, d_bounds, query_gest_bounds

    # --- Step 1: 动态图构建 ---
    print(f">>> [Retrieval] Reasoning Intent for: '{text}'")

    intent_graph = call_llm_intent(text)
    
    if not intent_graph:
        print("   [Warn] LLM failed to return graph.")
        return sample_indexes, d_bounds, query_gest_bounds

    intent_nodes = intent_graph.get("intent_nodes", [])
    
    # 准备对齐数据
    text_words_list = []
    text_ranges_list = []
    for t_item in text_times:
        text_words_list.append(t_item[1].lower())
        text_ranges_list.append(t_item[0])

    # 遍历每个意图节点
    for q_idx, node in enumerate(intent_nodes):
        semantic_label = node.get("semantic_label", node.get("original_text")) 
        anchor_word = node.get("source_anchor", node.get("original_text"))     
        synonyms = node.get("synonyms", [])
        visual_desc = node.get("visual_description", semantic_label)
        
        print(f"   [Node {q_idx}] Anchor: '{anchor_word}' -> Semantic: '{semantic_label}'")

        # ==========================================
        # 1. 时间对齐 (使用 source_anchor)
        # ==========================================
        matched_text_idx = -1
        clean_anchor = "".join([c for c in anchor_word.lower() if c.isalnum()])
        
        for idx, t_word in enumerate(text_words_list):
            clean_t = "".join([c for c in t_word if c.isalnum()])
            if clean_anchor == clean_t: 
                matched_text_idx = idx
                break
            elif clean_anchor in clean_t or clean_t in clean_anchor:
                if matched_text_idx == -1: matched_text_idx = idx
        
        if matched_text_idx == -1:
            print(f"       [Skip] Could not align anchor '{anchor_word}' to audio.")
            continue 

        query_start, query_end = text_ranges_list[matched_text_idx]
        query_gest_bounds[q_idx] = (anchor_word, "intent_node", query_start, query_end)

        # ==========================================
        # 2. 动作检索 Path A: 基于图谱的精准遍历
        # ==========================================
        search_terms = set([semantic_label.lower()] + [s.lower() for s in synonyms])
        graph_candidates = {} 
        graph_bounds_info = {}

        # 检查是否传入了图对象
        if kg_graph is None:
            print("       [Warn] No KG Graph provided, skipping Path A graph traversal.")
        else:
            # === 真正的图谱遍历逻辑 ===
            # 逻辑链: Keyword(Concept) --[入边]--> Semantic_Inst --[入边]--> Motion_Inst
            
            for keyword in search_terms:
                keyword_node_id = keyword.lower() # 假设图谱中概念节点 ID 为小写单词
                
                # 2.1 检查概念节点是否存在
                if not kg_graph.has_node(keyword_node_id):
                    continue
                
                # 2.2 找到提及该词的语义实例 (Semantic Instances)
                # 方向: Semantic_Inst -> MENTIONS -> Concept
                # 所以查 Concept 的 predecessors (入边)
                try:
                    semantic_instances = list(kg_graph.predecessors(keyword_node_id))
                except Exception:
                    continue # 防止图结构异常

                for sem_inst_id in semantic_instances:
                    # (可选) 检查边属性: if kg_graph[sem_inst_id][keyword_node_id]['label'] != 'MENTIONS': continue

                    # 2.3 找到对齐的动作实例 (Motion Instances)
                    # 方向: Motion_Inst -> ALIGNED_TO -> Semantic_Inst
                    # 所以查 Semantic_Inst 的 predecessors
                    try:
                        motion_instances = list(kg_graph.predecessors(sem_inst_id))
                    except Exception:
                        continue

                    for mot_inst_id in motion_instances:
                        # 2.4 提取动作元数据
                        if mot_inst_id not in kg_graph.nodes: continue
                        node_attrs = kg_graph.nodes[mot_inst_id]

                        # 还原 smp_idx (假设 ID 格式为 "Motion_Inst_{smp_idx}")
                        # 例如: "Motion_Inst_2_scott_0_10_10" -> "2_scott_0_10_10"
                        if not mot_inst_id.startswith("Motion_Inst_"): continue
                        smp_idx = mot_inst_id.replace("Motion_Inst_", "")

                        # 计算分数
                        base_score = 10 if keyword == semantic_label.lower() else 6
                        # 说话人奖励 (如果图谱存了 speaker 属性)
                        if node_attrs.get("speaker") == speaker_id: base_score += 3
                        
                        # 2.5 解析 word_timings 以获取精确时间
                        g_start, g_end = 0.0, 0.0
                        found_timing = False
                        
                        try:
                            # 从 GEXF 读取的属性通常是 JSON 字符串
                            word_timings_str = node_attrs.get("word_timings", "[]")
                            timings_list = json.loads(word_timings_str)
                            
                            for t in timings_list:
                                if t.get('word', '').lower() == keyword:
                                    g_start = float(t['start'])
                                    g_end = float(t['end'])
                                    found_timing = True
                                    break
                        except Exception as e:
                            # print(f"JSON Parse Error: {e}")
                            pass
                        
                        # 兜底：如果没找到具体单词时间，用切片时间
                        if not found_timing:
                            g_start = float(node_attrs.get("start_time", 0.0))
                            g_end = float(node_attrs.get("end_time", 0.0))

                        # 时域截取 (Temporal Clipping & Padding)
                        g_start = max(0.0, g_start - 0.2)
                        g_end = g_end + 0.5

                        # 记录候选
                        if smp_idx not in graph_candidates or base_score > graph_candidates[smp_idx]:
                            graph_candidates[smp_idx] = base_score
                            graph_bounds_info[smp_idx] = (keyword, "graph_match", g_start, g_end)

        # Path A 决策与精排
        path_a_hit = False
        final_idxs = []
        
        if len(graph_candidates) > 0:
            candidate_idxs = list(graph_candidates.keys())
            
            # 精排: 上下文语义相似度
            sorted_idxs = sort_sidx_by_textsimilarity(
                candidate_idxs, 
                visual_desc, 
                encoded_text, 
                text_feat_cache
            )
            final_idxs = sorted_idxs[:10]
            path_a_hit = True
            
            sample_indexes[q_idx] = final_idxs
            d_bounds[q_idx] = {}
            for idx in final_idxs:
                d_bounds[q_idx][idx] = graph_bounds_info[idx]
            
            print(f"       [Path A] Graph Hit! Found {len(final_idxs)} clips for '{semantic_label}'.")

        # ==========================================
        # 3. 向量检索兜底 (Path B - Fallback)
        # ==========================================
        if not path_a_hit:
            print(f"       [Path B] Fallback to Cross-Modal Vector Search: '{visual_desc}'")
            
            # ---------------------------------------------------------
            # 步骤 1: 编码 LLM 生成的动作描述 (Text Embedding)
            # ---------------------------------------------------------
            if text_model is None:
                print("       [Error] No text_model provided for vector search. Skipping.")
                continue

            with torch.no_grad():
                # 假设 text_model 接受 list[str] 并返回 tensor [1, Dim]
                # 根据您的具体模型 (TMR/MotionCLIP) API 调整此处调用方式
                # 例如: text_embedding = text_model.encode_text([visual_desc])
                # 这里假设传入的是一个能直接调用的 encoder
                try:
                    text_embedding = text_model([visual_desc]).to(device)
                except:
                    # 兼容某些模型需要 tokenize 的情况
                    text_embedding = text_model(visual_desc).to(device)
                
                # 归一化
                text_embedding = F.normalize(text_embedding, dim=-1)

            # ---------------------------------------------------------
            # 步骤 2: 从图谱中收集动作向量 (Motion Embeddings)
            # ---------------------------------------------------------
            candidate_ids = []
            motion_embeddings = []
            
            # 遍历图谱中的动作节点
            # 为了速度，建议在外部缓存这个 tensor，不要每次循环都重新构建
            if kg_graph is not None:
                for node_id, attrs in kg_graph.nodes(data=True):
                    # 筛选动作实例节点
                    # 检查节点 ID 格式或 type 属性
                    if not node_id.startswith("Motion_Inst_"): continue
                    
                    # 获取 clip_embedding
                    if "clip_embedding" not in attrs: continue
                    
                    try:
                        # GEXF 存储的数组通常是字符串或 JSON，需要解析
                        emb_data = attrs["clip_embedding"]
                        if isinstance(emb_data, str):
                            # 处理可能的格式: "[0.1, 0.2, ...]" 或 JSON
                            emb_vec = json.loads(emb_data)
                        else:
                            emb_vec = emb_data
                        
                        # 还原 smp_idx (用于 dataset 索引)
                        smp_idx = node_id.replace("Motion_Inst_", "")
                        
                        candidate_ids.append(smp_idx)
                        motion_embeddings.append(emb_vec)
                    except Exception as e:
                        continue

            if len(candidate_ids) == 0:
                print("       [Warn] No motion embeddings found in graph.")
                continue

            # ---------------------------------------------------------
            # 步骤 3: 计算相似度 (Cosine Similarity)
            # ---------------------------------------------------------
            # [N, Dim]
            motion_tensor = torch.tensor(motion_embeddings).to(device)
            motion_tensor = F.normalize(motion_tensor, dim=-1)
            
            # Similarity = Text_Emb @ Motion_Emb.T
            # [1, Dim] @ [Dim, N] -> [1, N]
            scores = torch.matmul(text_embedding, motion_tensor.T).squeeze(0)
            
            # ---------------------------------------------------------
            # 步骤 4: 选取 Top-K
            # ---------------------------------------------------------
            top_k = 10
            if len(scores) < top_k: top_k = len(scores)
            
            top_scores, top_indices = torch.topk(scores, k=top_k)
            
            final_idxs = []
            for i in range(top_k):
                idx_in_list = top_indices[i].item()
                smp_idx = candidate_ids[idx_in_list]
                final_idxs.append(smp_idx)

            # ---------------------------------------------------------
            # 步骤 5: 填入结果 (触发滑动窗口)
            # ---------------------------------------------------------
            sample_indexes[q_idx] = final_idxs
            d_bounds[q_idx] = {}
            for idx in final_idxs:
                # 标记 vector_search，触发 raggesture.py 中的滑动窗口逻辑
                d_bounds[q_idx][idx] = (visual_desc, "vector_search", 0.0, -1.0)
            
            if q_idx in query_gest_bounds:
                w, t, s, e = query_gest_bounds[q_idx]
                query_gest_bounds[q_idx] = (w, "vector_fallback", s, e)

    return sample_indexes, d_bounds, query_gest_bounds