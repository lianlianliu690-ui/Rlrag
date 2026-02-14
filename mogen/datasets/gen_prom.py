import os
import glob
import numpy as np
import pandas as pd
import librosa
import textgrid
from tqdm import tqdm

# ================= 配置 =================
ROOT_DIR = "/Dataset/mas-liu.lianlian/beat_v2.0.0/beat_english_v2.0.0"
WAV_FOLDER = os.path.join(ROOT_DIR, "wave16k")
TEXTGRID_FOLDER = os.path.join(ROOT_DIR, "textgrid")
OUTPUT_FOLDER = os.path.join(ROOT_DIR, "prom")
# =======================================

def calculate_acoustic_features(wav_path, start_t, end_t):
    """计算片段内的声学特征：最大音高、最大能量、时长"""
    # 加载音频片段太慢，建议在主循环加载一次，这里简化处理
    # 为了速度，实际操作应该在外部加载整个 wav

    return 0, 0 # 占位，逻辑在下面主函数实现

def main():
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    
    tg_files = glob.glob(os.path.join(TEXTGRID_FOLDER, "*.TextGrid"))
    
    for tg_path in tqdm(tg_files):
        basename = os.path.splitext(os.path.basename(tg_path))[0]
        wav_path = os.path.join(WAV_FOLDER, f"{basename}.wav")
        save_path = os.path.join(OUTPUT_FOLDER, f"{basename}.prom")
        
        if not os.path.exists(wav_path): continue
        
        # 1. 加载音频
        y, sr = librosa.load(wav_path, sr=16000)
        
        # 2. 提取 F0 (音高) 和 RMS (能量)
        # f0: (time_steps,)
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
        rms = librosa.feature.rms(y=y)[0]
        times = librosa.times_like(rms, sr=sr)
        
        # 处理 F0 中的 NaN (无声段)
        f0 = np.nan_to_num(f0)

        # 3. 读取 TextGrid
        tg = textgrid.TextGrid.fromFile(tg_path)
        word_tier = tg[0] # 假设第一层是 words

        # 收集所有词的特征，用于计算相对突显度
        words_data = []
        for interval in word_tier:
            text = interval.mark.strip()
            if not text or text.lower() in ["<sil>", ""]: continue
            
            start = interval.minTime
            end = interval.maxTime
            duration = end - start
            
            # 找到对应的时间帧索引
            idx_start = np.searchsorted(times, start)
            idx_end = np.searchsorted(times, end)
            
            if idx_end > idx_start:
                word_f0 = f0[idx_start:idx_end]
                word_rms = rms[idx_start:idx_end]
                max_f0 = np.max(word_f0) if len(word_f0) > 0 else 0
                max_rms = np.max(word_rms) if len(word_rms) > 0 else 0
            else:
                max_f0 = 0
                max_rms = 0
            
            words_data.append({
                "basename": basename,
                "start": start,
                "end": end,
                "word": text,
                "duration": duration,
                "max_f0": max_f0,
                "max_rms": max_rms
            })
            
        if not words_data: continue

        # 4. 计算突显度 (Prominence)
        # 规则：如果一个词的 音高 或 能量 显著高于句子的平均水平，标记为 Prominence
        df = pd.DataFrame(words_data)
        
        # 计算 Z-score (标准分数)
        df['f0_z'] = (df['max_f0'] - df['max_f0'].mean()) / (df['max_f0'].std() + 1e-6)
        df['rms_z'] = (df['max_rms'] - df['max_rms'].mean()) / (df['max_rms'].std() + 1e-6)
        df['dur_z'] = (df['duration'] - df['duration'].mean()) / (df['duration'].std() + 1e-6)
        
        # 综合得分：音高权重最高，能量次之，时长最后
        df['score'] = 0.5 * df['f0_z'] + 0.3 * df['rms_z'] + 0.2 * df['dur_z']
        
        # 设定阈值 (例如超过 0.5 个标准差就算重音)
        df['prominence'] = df['score'].apply(lambda x: "True" if x > 0.5 else "NA")
        
        # 5. 计算边界 (Boundary)
        # 简单逻辑：看停顿时间
        df['next_start'] = df['start'].shift(-1)
        df['gap'] = df['next_start'] - df['end']
        df['boundary'] = df['gap'].apply(lambda x: "Boundary" if x > 0.1 else "NA")
        df.iloc[-1, df.columns.get_loc('boundary')] = "Boundary" # 最后一个词必是边界

        # 6. 保存
        df_save = df[["basename", "start", "end", "word", "prominence", "boundary"]]
        df_save.to_csv(save_path, sep="\t", header=False, index=False)

if __name__ == "__main__":
    main()