import os
import glob
import json
import textgrid
import openai
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# ==================== 配置区域 ====================
dotenv_path = "/home/mas-liu.lianlian/RAG-Gesture/.env"
load_dotenv(dotenv_path)

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com") 

TEXTGRID_FOLDER = "/Dataset/mas-liu.lianlian/beat_v2.0.0/beat_english_v2.0.0/textgrid"
OUTPUT_FOLDER = "/Dataset/mas-liu.lianlian/beat_v2.0.0/beat_english_v2.0.0/discourse_rels"

MODEL_NAME = "deepseek-chat" 
# =================================================

print(f"🚀 使用模型: {MODEL_NAME}")
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def extract_tokens_from_textgrid(tg_path):
    """
    提取 TextGrid 内容，并展平成一个单一的 Token 列表，供后续查索引使用。
    """
    try:
        tg = textgrid.TextGrid.fromFile(tg_path)
    except Exception as e:
        print(f"❌ TextGrid Error: {tg_path}")
        return "", []
    
    word_tier = None
    possible_names = ["words", "word", "transcript", "MAU"]
    for tier in tg:
        if tier.name.lower() in possible_names:
            word_tier = tier
            break
    if word_tier is None and len(tg) > 0: word_tier = tg[0]
        
    full_text_list = []
    tokens_list = [] 
    
    # 这里的 tokens_list 对应代码里的 all_tokens
    if word_tier:
        for idx, interval in enumerate(word_tier):
            text = interval.mark.strip()
            # 过滤掉静音，但要小心，如果索引错位可能需要保留空token
            # 这里我们假设过滤掉静音后，生成的文本与GPT理解的一致
            if text and text.lower() not in ["<sil>", "<p>", ""]: 
                full_text_list.append(text)
                tokens_list.append({
                    "surface": text,      
                    "startSec": interval.minTime,
                    "endSec": interval.maxTime
                })
            
    return " ".join(full_text_list), tokens_list

def clean_json_string(json_str):
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].split("```")[0]
    elif "```" in json_str:
        json_str = json_str.split("```")[1].split("```")[0]
    return json_str.strip()

def get_relations_from_gpt(text):
    system_prompt = """
    Extract discourse relations (PDTB style).
    Return JSON object: {"relations": [{"connective": "...", "sense": "...", "arg1": "...", "arg2": "..."}]}
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"}, 
            temperature=0.1
        )
        result = clean_json_string(response.choices[0].message.content)
        data = json.loads(result)
        
        if isinstance(data, list): return data
        if isinstance(data, dict): return data.get("relations", [])
        return []
    except:
        return []

def find_token_indices(target_text, all_tokens, start_search_idx=0):
    """
    核心函数：根据文本内容，去 all_tokens 列表里找到对应的 [索引号列表]
    例如: target="for example", tokens=[..., "for"(idx 5), "example"(idx 6), ...]
    返回: [5, 6]
    """
    if not target_text: return []
    
    target_words = target_text.split()
    target_len = len(target_words)
    if target_len == 0: return []

    def clean(s): return s.lower().strip(".,?!\"'")

    # 滑动窗口搜索
    for i in range(start_search_idx, len(all_tokens) - target_len + 1):
        match = True
        for j in range(target_len):
            if clean(target_words[j]) not in clean(all_tokens[i+j]["surface"]):
                match = False
                break
        
        if match:
            # 找到了！返回对应的索引列表 [i, i+1, i+2...]
            return list(range(i, i + target_len))
            
    return []

def main():
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    files = glob.glob(os.path.join(TEXTGRID_FOLDER, "*.TextGrid"))
    
    # files = files[0:10] # 测试用

    print(f"Processing {len(files)} files...")
    for tg_path in tqdm(files):
        base_name = os.path.splitext(os.path.basename(tg_path))[0]
        save_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_whisper_relations.json")
        
        # 1. 提取 tokens
        full_text, tokens = extract_tokens_from_textgrid(tg_path)
        
        if not full_text:
            empty_struct = {"sentences": [{"tokens": []}], "relations": []}
            with open(save_path, 'w') as f: json.dump(empty_struct, f)
            continue

        # 2. GPT 获取文本关系
        gpt_rels = get_relations_from_gpt(full_text)
        
        # 3. 构造 PDTB 格式的 relation
        pdtb_relations = []
        
        for rel in gpt_rels:
            conn_text = rel.get("connective", "")
            arg1_text = rel.get("arg1", "")
            arg2_text = rel.get("arg2", "")
            sense = rel.get("sense", "Contingency.Cause")

            # 查找索引 (TokenList)
            # 这里简化处理：每次都从头找。更严谨的逻辑可能需要记录上次找到的位置，但对短文本通常够用。
            conn_indices = find_token_indices(conn_text, tokens)
            arg1_indices = find_token_indices(arg1_text, tokens)
            arg2_indices = find_token_indices(arg2_text, tokens)

            # 必须要有连接词的索引，否则这条关系没法用
            if conn_indices:
                pdtb_item = {
                    "Connective": {
                        "RawText": conn_text,
                        "TokenList": conn_indices
                    },
                    "Arg1": {
                        "TokenList": arg1_indices
                    },
                    "Arg2": {
                        "TokenList": arg2_indices
                    },
                    "Sense": [sense] # 必须是列表 ["Contingency.Cause"]
                }
                pdtb_relations.append(pdtb_item)

        # 4. 组装最终 JSON
        final_output = {
            "sentences": [
                {
                    "tokens": tokens # 原始 tokens 列表
                }
            ],
            "relations": pdtb_relations
        }

        with open(save_path, 'w') as f:
            json.dump(final_output, f, indent=4)

if __name__ == "__main__":
    main()