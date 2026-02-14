import torch
import numpy as np
# from huggingface_hub import InferenceClient
import json
import glob
import re
import copy
import time
import random
import networkx as nx

from .utils import sort_sidx_by_textsimilarity, get_word_similarity_score, map_conns_to_prominence


from dotenv import load_dotenv

import torch.nn.functional  as F

from openai import OpenAI
import os

hf_infclient = None
# hf_infclient = InferenceClient(
#     "meta-llama/Meta-Llama-3-8B-Instruct",
#     token="YOUR_TOKEN_HERE",
# )

# ================= 1. 更新 Prompt =================
INTENT_PROMPT = """
### Role
You are a Motion Director Agent. Your goal is to analyze user speech and construct a **Dynamic Intent Graph** to guide 3D character animation.

### Input
You will be provided with:
1. **Full Transcript**: The complete speech text.
2. **Focus Segment**: The specific sentence(s) we are currently animating.

### Goal
Analyze the **Focus Segment** to construct a Dynamic Intent Graph. 
Use the **Full Transcript** to infer:
1. **Global Emotion**: The underlying mood of the speaker.
2. **Reference Resolution**: Identify who 'he', 'it', 'that' refers to.
3. **Discourse Logic**: Identify logical relationships (Contrast, Causality, Sequence) between the Focus Segment and its context.


### Discourse Logic Rules (Crucial)
- **CONTRAST (BUT/HOWEVER)**: If the context shows a conflict (e.g., "I was sad, BUT I smiled"), the visual description must reflect this conflict (e.g., "forced smile, sad eyes").
- **CAUSALITY (BECAUSE/SO)**: If the current action is a reaction to a previous event, the intensity should match the cause.
- **SEQUENCE (THEN)**: Ensure the motion flows naturally from the previous context.

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
    // Capture discourse relations if they exist within the Focus Segment or linking to context
    {"source": "n0", "target": "n1", "relation": "BUT"}
  ]
}
"""

load_dotenv()
openai_api_key = os.getenv("Qwen_API_KEY")
client = OpenAI(api_key=openai_api_key,base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

def call_llm_json(text, context_to_send,model="qwen-plus"):
    """
    [修改] 调用 Qwen 模型并强制 JSON 输出
    """
    if not client.api_key:
        print("[LLM Error] Client not initialized with API Key.")
        return {"target_emotion": "neutral", "intent_nodes": []}
    
    user_msg = f"""
    [Full Transcript]:
    "{context_to_send}"
    
    [Focus Segment]:
    "{text}"
    
    Please analyze the [Focus Segment] based on the context of [Full Transcript].
    """
    try:
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": INTENT_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            response_format={"type": "json_object"} 
        )
        content = completion.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"[LLM Error] Qwen call failed: {e}")
        # 返回空结构防止报错
        return {"target_emotion": "neutral", "intent_nodes": []}

def parse_smp_name_from_id(node_id):
    """
    从图谱节点 ID 解析出 Dataset 中的 sample_name
    假设 ID 格式: "Motion_Inst_{sample_name}" 或包含其他后缀
    """
    if node_id.startswith("Motion_Inst_"):
        id = node_id.replace("Motion_Inst_", "")
        parts = id.split("_")
        clean_id = "_".join(parts[:-1]) + "/" + parts[-1]
        return clean_id
    return node_id

def llm_retrieval(
    text,
    text_times,
    speaker_id,
    prominence,
    db_idx_2_gesture_labels,
    db_idx_2_prominence,
    encoded_text,
    text_feat_cache,
    kg_graph=None,       # [新增] 静态图谱
    name_to_idx=None,    # [新增] ID映射
    text_model=None, # [新增] 用于 Path B
    motion_model=None,
    dataset_handle=None, # [新增] 用于读取原始动作数据 (例如 self.dataset)
    device='cpu',
    cached_embeds=None,  # [新增] 接收 Tensor [N, Dim]
    cached_idxs=None,    # [新增] 接收 Tensor [N]
    full_text=None,
    focus_window=None,
    idx_2_name = None,
):
    """
    混合检索主入口：动态图构建 (Qwen) -> 图谱遍历 (Path A) -> 向量兜底 (Path B)
    """
    sample_indexes = {} #存储 检索到的候选动作 ID 列表
    d_bounds = {}       #存储 检索到的动作在数据库原始文件中的具体信息
    query_gest_bounds = {} #存储 用户当前输入的文本 中，每一个需要做动作的词的时间戳和检索来源的可信度标签

    if not text.strip():
        return sample_indexes, d_bounds, query_gest_bounds

    # --- Step 1: 动态图构建 (Intent Reasoning with Qwen) ---
    print(f">>> [Retrieval] Qwen Reasoning for: '{text}'")
    # 1. 如果没有全文，回退到只用 text
    context_to_send = full_text if full_text else text
    

    
    # 调用 Qwen
    intent_json = call_llm_json(text, context_to_send,model="qwen-plus")
    
    target_emotion = intent_json.get("target_emotion", "neutral")
    intent_nodes = intent_json.get("intent_nodes", [])
    
    # 建立单词到时间戳的查找表
    text_words_list = [t[1].lower() for t in text_times]
    text_ranges_list = [t[0] for t in text_times]

    # 遍历每个意图节点
    for q_idx, node in enumerate(intent_nodes):
        semantic_label = node.get("semantic_label", "")
        anchor_word = node.get("source_anchor", semantic_label)
        synonyms = node.get("synonyms", [])
        visual_desc = node.get("visual_description", "")
        
        # 1.1 时间对齐
        matched_idx = -1
        clean_anchor = "".join([c for c in anchor_word.lower() if c.isalnum()])
        for i, t_word in enumerate(text_words_list):
            clean_t = "".join([c for c in t_word if c.isalnum()])
            if clean_anchor in clean_t or clean_t in clean_anchor:
                matched_idx = i
                break
        
        if matched_idx == -1:
            print(f"   [Skip] Anchor '{anchor_word}' not found in audio.")
            continue
            
        q_start, q_end = text_ranges_list[matched_idx]
        query_gest_bounds[q_idx] = (anchor_word, "intent_node", q_start, q_end)

        keywords = set([semantic_label.lower()] + [s.lower() for s in synonyms])

        # =========================================================
        # === Path A: 基于图谱的精准检索 (Graph Grounding) ===
        # =========================================================
        path_a_candidates = {} 
        path_a_hit = False

        if kg_graph and name_to_idx:
            for kw in keywords:
                if not kg_graph.has_node(kw): continue
                
                try:
                    # 拓扑遍历: Concept <-[MENTIONS]- Semantic（语义实例节点） <-[ALIGNED_TO]- Motion（动作实例节点）
                    sem_insts = list(kg_graph.predecessors(kw))
                    for sem_inst in sem_insts:
                        mot_insts = list(kg_graph.predecessors(sem_inst))
                        for mot_inst in mot_insts:
                            attrs = kg_graph.nodes[mot_inst]
                            smp_name = parse_smp_name_from_id(mot_inst)
                            if smp_name not in name_to_idx: continue
                            smp_idx_int = name_to_idx[smp_name]

                            score = 1.0
                            if kw == semantic_label.lower(): score += 5.0
                            if attrs.get("emotion_tag") == target_emotion: score += 2.0

                            # [修改] 移到这里的粗排逻辑：说话人风格奖励
                            try:
                                # 假设 smp_name 格式: "2_scott_..." -> ID=2
                                # 注意：speaker_id 通常是 0-indexed，文件名可能是 1-indexed，需确认您的数据格式
                                # 这里假设文件名 ID = speaker_id + 1
                                candidate_spk_id = int(smp_name.split('_')[0])
                                if candidate_spk_id == int(speaker_id + 1): 
                                    score += 1.5
                            except:
                                pass

                            # 1. 获取片段的绝对起止时间 (用于计算时长)
                            clip_abs_start = float(attrs.get("start_time", 0.0))
                            clip_abs_end = float(attrs.get("end_time", 0.0))
                            
                            # 解析 word_timings
                            timings = json.loads(attrs.get("word_timings", "[]"))
                            g_start, g_end = 0.0, 0.0
                            found_t = False
                            clean_kw = "".join([c for c in kw.lower() if c.isalnum()])
                            for t in timings:
                                # [优化] 预处理数据库里的词
                                clean_db_word = "".join([c for c in t['word'].lower() if c.isalnum()])
                                
                                ## 检查策略 (精确匹配 or 包含匹配)
                                is_match = (clean_db_word == clean_kw)
                                if not is_match and len(clean_kw) > 2 and len(clean_db_word) > 2:
                                     if clean_kw in clean_db_word or clean_db_word in clean_kw:
                                         is_match = True
                                
                                if is_match:
                                    # [情况 A] 找到特定词
                                    # 根据截图，word_timings 里的 start (0.36) 已经是【相对时间】了
                                    # 所以直接取值即可，不需要减去 clip_abs_start
                                    g_start = t['start']
                                    g_end = t['end']
                                    found_t = True
                                    break   
                            
                            if not found_t:
                                # [情况 B] 没找到词，回退到整段动作 (Fallback)
                                # [修改前] g_start = clip_abs_start (返回了 31.33，是绝对时间 ❌)
                                # [修改后] 返回相对于该片段的时间 (从 0.0 开始 ✅)
                                g_start = 0.0
                                g_end = max(0.0, clip_abs_end - clip_abs_start)

                            # # Padding
                            # g_start = max(0.0, g_start - 0.2)
                            # g_end = g_end + 0.5
                            
                            if smp_idx_int not in path_a_candidates or score > path_a_candidates[smp_idx_int]['score']:
                                path_a_candidates[smp_idx_int] = {
                                    'score': score,
                                    'word': kw,
                                    'start': g_start,
                                    'end': g_end
                                }
                except Exception as e:
                    # print(f"Graph traversal error: {e}")
                    continue

        # Path A 精排
        if len(path_a_candidates) > 0:
            # 1. 从 kwargs 中获取反向映射 (如果没有传，给空字典防止报错)
            idx_to_name = idx_2_name
            
            # 2. 调用新的精排函数
            # 注意：第一个参数传的是 path_a_candidates 整个字典
            final_idxs = rank_candidates_with_proxy(
                path_a_candidates,   # 传字典
                text_feat_cache,     # 传缓存
                encoded_text,        # 传 Query 向量
                idx_to_name,         # 传 ID->Name 映射
                device,              # 传 device (如 'cuda')
                target_speaker_id=speaker_id, # [传参] 传入当前目标说话人
                search_radius=20     # 可选：搜索半径
            )
            
            # 3. 填入结果
            sample_indexes[q_idx] = final_idxs
            d_bounds[q_idx] = {}
            for idx in final_idxs:
                # 无论是否借用了邻居的分数，这里依然记录原始候选的信息
                info = path_a_candidates[idx]
                d_bounds[q_idx][idx] = (
                    info['word'], 
                    "graph_match", 
                    info['start'], 
                    info['end'],
                    # visual_desc            # 5. [新增] 视觉描述 (用于Adapter Guidance)
                )
            
            path_a_hit = True
            print(f"   [Path A] Graph Hit! Re-ranked {len(path_a_candidates)} -> {len(final_idxs)} items using Proxy.")

        # =========================================================
        # === Path B: 向量化检索 (Vector Retrieval) ===
        # =========================================================
        if not path_a_hit and text_model and cached_embeds is not None:
            # 1. 编码文本 Query
            visual_desc_vec = text_model([visual_desc]) 
            visual_desc_vec = F.normalize(visual_desc_vec, p=2, dim=1)
            
            # 2. 粗排 (Coarse Retrieval) - 找文件
            scores = torch.matmul(visual_desc_vec, cached_embeds.T).squeeze(0)
            topk_vals, topk_inds = torch.topk(scores, k=10) # 找前10个备选
            
            final_idxs = cached_idxs[topk_inds].cpu().tolist()
            
            # 3. 填入结果 & 精排 (Fine-grained Sliding Window)
            sample_indexes[q_idx] = final_idxs
            d_bounds[q_idx] = {}
            
            for rank, idx in enumerate(final_idxs):
                # 默认值
                s_time, e_time = 0.0, -1.0 
                
                # 【新增】只对 Top-1 (或 Top-3) 做精细的时间定位
                # 避免对所有 10 个候选都做，太慢
                if rank == 0 and motion_model and dataset_handle:
                    try:
                        # 3.1 加载原始动作
                        # 注意：这里需要你能通过 ID 拿到原始数据
                        raw_data = dataset_handle[idx]["motion_h3d"] # [T, D]
                        
                        # 3.2 语义滑动窗口
                        s_time, e_time = find_semantic_window(
                            motion_model, 
                            raw_data, 
                            visual_desc_vec, # 使用对齐的文本向量
                            fps=15, 
                            device=device
                        )
                        # print(f"Refined Window for {idx}: {s_time:.2f}-{e_time:.2f}")
                    except Exception as e:
                        print(f"Window search failed: {e}")
                        s_time, e_time = 0.0, -1.0 # 回退

                # 存入 d_bounds
                # 注意：这里存入的是精确时间，类型依然叫 vector_fallback
                # 或者您可以起个新名字叫 "vector_refined"
                d_bounds[q_idx][idx] = (
                    visual_desc, 
                    "vector_fallback", 
                    round(s_time, 3), 
                    round(e_time, 3), # 四舍五入，保留三位小数
                    # visual_desc            # 5. [新增] 视觉描述
                )
                

    return sample_indexes, d_bounds, query_gest_bounds

def rank_candidates_with_proxy(
    path_a_candidates, 
    text_feat_cache, 
    encoded_text, 
    idx_to_name, 
    device,
    target_speaker_id, # [新增] 传入目标说话人ID (int)
    search_radius=20
):
    """
    [Path A 精排 - 终极融合版]
    融合了: Graph语义 + BERT语境 + 说话人风格(原项目) + 精确词匹配(原项目)
    """
    candidate_idxs = list(path_a_candidates.keys())
    
    # 1. 获取代理映射 (Int_ID -> String_Key)
    proxy_map = get_proxy_map(candidate_idxs, text_feat_cache, idx_to_name, search_radius)
    
    final_results = []
    
    # 归一化 Query 向量
    text_feature = F.normalize(encoded_text.to(device), p=2, dim=1)

    for idx in candidate_idxs:
        # === 基础分 (Graph Score) ===
        # 包含了: Keyword Match (+5), Emotion Match (+2)
        base_score = path_a_candidates[idx]['score']
        

        # === [新增] 精确词匹配奖励 (借鉴原项目) ===
        # 检查这个候选是不是完全匹配了 LLM 的原始意图词
        # path_a_candidates[idx]['word'] 是图谱里的词
        # 我们很难直接拿到 User Query Word，但可以依靠 graph_score 的来源判断
        # 如果 graph_score 很高(>5)，说明大概率是原词匹配，这里不再重复加分
        
        # === 语境分 (Context Score) ===
        context_score = 0.0
        
        if idx in proxy_map:
            cache_key = proxy_map[idx]
            raw_text_f, _ = text_feat_cache[cache_key]
            
            # 归一化 Key 向量
            text_f_norm = F.normalize(raw_text_f.to(device), p=2, dim=1)
            
            # 计算 Cosine Similarity [-1, 1]
            text_sim_matrix = torch.mm(text_feature, text_f_norm.T)
            text_sim = torch.diagonal(text_sim_matrix).mean().item()
            
            # 加权融合: 语境分权重 10.0
            context_score = text_sim * 10.0
            
            # # [惩罚跨样本] 如果是借来的邻居，稍微扣一点分(0.5)，优先选原本就有缓存的
            # if str(idx) not in cache_key: # 简单判断 ID 是否在 Key 里 (近似)
            #      # 或者比较 idx_to_name[idx] 和 cache_key
            #      pass 
        else:
            # 没缓存，严重惩罚
            context_score = -5.0
            
        # === 最终得分 ===
        final_score = base_score + context_score
        final_results.append((idx, final_score))
            
    # 排序
    final_results.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in final_results[:10]]

def get_proxy_map(candidate_idxs, text_feat_cache, idx_to_name, search_radius=20):
    """
    为候选样本寻找有效的缓存代理。
    适配格式: idx_to_name[0] = '2_scott_1_10_10/0'
    """
    proxy_map = {} # {Original_ID : String_Cache_Key}
    
    # 辅助函数：提取序列前缀，用于同源校验
    # 输入: '2_scott_1_10_10/0'
    # 目标: 提取 '2_scott_1' (Speaker_Name_Session) 确保不跨 Session 借用
    def get_seq_prefix(full_name):
        if not full_name: return ""
        # 1. 去掉后缀 /0
        base = full_name.split('/')[0] #2_scott_1_10_10
        
        return base

    for idx in candidate_idxs:
        # Step 0: 获取当前样本名 (例如 '2_scott_1_10_10/0')
        curr_name = idx_to_name.get(idx)
        if not curr_name: continue
        
        # Step 1: 【直接匹配】
        # 因为 curr_name 已经是完整格式，直接查 Cache
        if curr_name in text_feat_cache:
            proxy_map[idx] = curr_name
            continue
            
        # Step 2: 【邻居借用】(处理被采样跳过的数据)
        curr_prefix = get_seq_prefix(curr_name)
        
        found = False
        for offset in range(1, search_radius + 1):
            for neighbor_idx in [idx - offset, idx + offset]:
                neighbor_name = idx_to_name.get(neighbor_idx)
                if not neighbor_name: continue
                
                # 同源校验: 必须是同一个 Session 下的动作
                if get_seq_prefix(neighbor_name) != curr_prefix:
                    continue
                
                # 查缓存
                if neighbor_name in text_feat_cache:
                    proxy_map[idx] = neighbor_name # 借用邻居的完整 Key
                    found = True
                    break
            if found: break
            
    return proxy_map

def find_semantic_window(motion_model, raw_motion, text_vec, fps=15, device='cpu'):
    """
    语义滑动窗口：在原始动作上滑动，编码后与文本向量比对。
    """
    # raw_motion: [T, Joint_Dim]
    # text_vec: [1, Dim] (已经归一化)
    
    # 定义窗口参数
    scales = [1.5, 3.0] # 搜索 1.5秒 和 3秒 的窗口
    best_score = -1.0
    best_window = (0.0, raw_motion.shape[0] / fps) # 默认全长

    # 将动作移到 GPU
    raw_motion = raw_motion.to(device)
    
    with torch.no_grad():
        for duration in scales:
            win_len = int(duration * fps)
            stride = int(0.5 * fps)
            
            if win_len >= raw_motion.shape[0]: continue

            # 为了加速，可以把所有窗口拼成一个 Batch 一次性编码
            windows_list = []
            starts = []
            
            for i in range(0, raw_motion.shape[0] - win_len, stride):
                # 切片
                chunk = raw_motion[i : i+win_len]
                # 某些 Encoder 需要固定长度，这里假设它是变长的或者内部会处理
                windows_list.append(chunk)
                starts.append(i)
            
            if not windows_list: continue
            
            # Pad & Stack (如果 Encoder 支持 Batch 处理)
            # 这里简单起见，假设 pad_sequence 处理
            from torch.nn.utils.rnn import pad_sequence
            batch_windows = pad_sequence(windows_list, batch_first=True, padding_value=0)
            
            # --- 关键点：调用动作编码器 ---
            # 假设 motion_model.encode 返回 [Batch, Dim]
            motion_embs = motion_model(batch_windows, lengths=[win_len]*len(windows_list))
            motion_embs = F.normalize(motion_embs, p=2, dim=1)
            
            # 计算相似度: [Batch]
            scores = torch.matmul(motion_embs, text_vec.T).squeeze(1)
            
            # 找最大值
            curr_best_val, curr_best_idx = torch.max(scores, dim=0)
            
            if curr_best_val > best_score:
                best_score = curr_best_val.item()
                s_idx = starts[curr_best_idx]
                best_window = (s_idx / fps, (s_idx + win_len) / fps)
    
    return best_window