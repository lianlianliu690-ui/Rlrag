import os
import sys
import torch
import numpy as np
# --- 补丁开始 ---
if not hasattr(np, 'float'):
    np.float = float
# --- 补丁结束 ---
import networkx as nx
import yaml
import json
import argparse
from argparse import Namespace
from tqdm import tqdm
from mmcv import Config

# --- 补丁开始: 修复 models.motionclip 导入问题 ---
import sys
import os

# 1. 强制将 SynTalker 根目录加入 Python 搜索路径的最前面
SYNTALKER_ROOT = "/Dataset4D/public/mas-liu.lianlian/code/SynTalker"
MODELS_DIR = os.path.join(SYNTALKER_ROOT, "models")

print(f"DEBUG: Adding path {SYNTALKER_ROOT}")
# 无论如何都插到第一个，确保优先级最高，防止同名包冲突
if SYNTALKER_ROOT not in sys.path:
    sys.path.insert(0, SYNTALKER_ROOT)

if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR) # 🔥 这一步解决了 'No module named temos'

from models.temos.motionencoder.actor import ActorAgnosticEncoder

print("DEBUG: ✅ MotionCLIP Imported Successfully!")
HAS_TMR = True
# --- 补丁结束 ---
import os
import sys
import warnings

# 添加必要的路径到Python路径
sys.path.insert(0, '/home/mas-liu.lianlian/RLrag')
from mogen.models.transformers.gesture_vae import TransformerVAE
from mogen.datasets.builder import build_dataset, build_dataloader
from mogen.models.utils import rotation_conversions as rc

# ================= 1. TMR Encoder Wrapper (新版) =================
class TMRMotionWrapper:
    """
    针对 TMR (Text-Motion-Retrieval) 模型的封装
    自动处理 Config 缺失和分离权重加载问题
    """
    def __init__(self, model_dir, device):
        self.device = device
        print(f">>> [Loader] Loading TMR (ActorAgnosticEncoder) from: {model_dir}")

        # 1. 实例化模型 (参数照抄 SynTalker)
        # 注意：SynTalker 硬编码了 nfeats=623, vae=True, num_layers=4
        # 如果您的模型也是这一套权重 (motion_epoch=299.ckpt)，请保持一致
        self.model = ActorAgnosticEncoder(nfeats=623, vae=True, num_layers=4)
        
        self.model.eval()
        self.model.to(device)

        # 2. 加载权重
        # 自动寻找 .ckpt 文件
        ckpt_path = os.path.join(model_dir, "motion_epoch=299.ckpt") 
        # 或者遍历寻找
        if not os.path.exists(ckpt_path):
             import glob
             ckpts = glob.glob(os.path.join(model_dir, "motion_*.ckpt"))
             if ckpts: ckpt_path = ckpts[0]
        
        if os.path.exists(ckpt_path):
            print(f"    Loading weights from: {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location=device)
            self.model.load_state_dict(state_dict)
        else:
            print(f"❌ [Error] Ckpt not found in {model_dir}")

        # 4. 均值方差加载 (Mean/Std Logic)
        self.mean = None; self.std = None
        # 尝试常用名
        for name in ["mean.npy", "beatx_1-30_amass_h3d_mean.npy"]:
            p = os.path.join(model_dir, name)
            if os.path.exists(p):
                self.mean = torch.from_numpy(np.load(p)).to(device).float()
                break
        for name in ["std.npy", "beatx_1-30_amass_h3d_std.npy"]:
            p = os.path.join(model_dir, name)
            if os.path.exists(p):
                self.std = torch.from_numpy(np.load(p)).to(device).float()
                break
        
        if self.mean is None: print("!! [CRITICAL] No mean.npy found. Normalization disabled!")

    def get_motion_embeddings(self, raw_motion, lengths):
        with torch.no_grad():
            motions = raw_motion.to(self.device).float()
            # 归一化
            if self.mean is not None and self.std is not None:
                motions = (motions - self.mean) / (self.std + 1e-8)
            
            dist = self.model(motions, lengths)
            return dist.loc
    
    def __call__(self, motion_data):
        """
        使TMRMotionWrapper对象可调用
        接受运动数据，返回嵌入特征
        """
        # 假设motion_data是[B, T, D]格式
        batch_size, seq_len, feat_dim = motion_data.shape
        lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=self.device)
        return self.get_motion_embeddings(motion_data, lengths)

# ================= 2. 主构建器 =================
class MotionInstanceBuilder:
    def __init__(self, upper_cfg_path, hands_cfg_path, dataset_cfg_path, device='cuda'):
        self.device = device
        self.output_dir = "/Dataset4D/public/mas-liu.lianlian/output/RAGesture/rl_kg/motion_instances"
        self.motion_assets_dir = os.path.join(self.output_dir, "assets")
        
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)
        if not os.path.exists(self.motion_assets_dir): os.makedirs(self.motion_assets_dir)
        
        self.upper_cfg_path = upper_cfg_path
        self.hands_cfg_path = hands_cfg_path
        self.dataset_cfg_path = dataset_cfg_path
        self.graph = nx.DiGraph()

    def _load_single_vae(self, config_path):
        print(f"Loading VAE from {config_path}...")
        with open(config_path, "r") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        args = Namespace(**cfg)
        model = TransformerVAE(args).to(self.device)
        model.eval()
        
        ckpt_path = args.test_ckpt
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(os.path.dirname(config_path), ckpt_path)
        
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location=self.device)
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.get('model_state', state_dict).items()}
            model.load_state_dict(new_state_dict, strict=False)
        return model

    def _load_motion_encoder(self):
            # 1. 定义您的 TMR 模型绝对路径
            # 请确认这个路径下有 config.yaml (可选) 和 .ckpt 文件
            TMR_DIR = "/Dataset4D/public/mas-liu.lianlian/pretrained_models/beatx_1-30_amass_h3d_tmr/"
            
            print(f">>> [Loader] Attempting to load TMR from: {TMR_DIR}")
            
            # 2. 物理检查
            if not os.path.exists(TMR_DIR):
                print(f"❌ [Error] Path not found: {TMR_DIR}")
                return None

            # 3. 尝试加载
            try:
                # 确保 HAS_TMR 为 True (在文件头部被 import 逻辑设置)
                if not globals().get('HAS_TMR', False):
                    print("❌ [Error] HAS_TMR is False. MotionCLIP import failed previously.")
                    return None

                # 实例化 Wrapper
                encoder = TMRMotionWrapper(TMR_DIR, self.device)
                print("✅ [Loader] TMR Motion Encoder loaded successfully!")
                return encoder
                
            except Exception as e:
                print(f"❌ [Critical Error] Failed to load TMR: {e}")
                import traceback
                traceback.print_exc()
                return None

    def _load_dataloader(self, config_path):
        print(f"Loading Dataset from {config_path}...")
        cfg = Config.fromfile(config_path)
        # 🔥 [新增] 获取 FPS，用于计算时间秒数
        # 默认 15 或 30，具体看 config
        self.fps = cfg.get('pose_fps', 15) 
        if hasattr(cfg.data, 'train') and hasattr(cfg.data.train, 'fps'):
             self.fps = cfg.data.train.fps
        print(f">>> Dataset FPS: {self.fps}")
        return build_dataloader(build_dataset(cfg.data.train), samples_per_gpu=1, workers_per_gpu=4, dist=False, shuffle=False)

    def _preprocess_motion(self, motion_data):
        bs, n, j_raw = motion_data.shape
        num_joints = j_raw // 3
        motion_mat = rc.axis_angle_to_matrix(motion_data.reshape(bs, n, num_joints, 3))
        motion_6d = rc.matrix_to_rotation_6d(motion_mat).reshape(bs, n, num_joints * 6)
        return motion_6d

    def build(self):
        self.upper_vae = self._load_single_vae(self.upper_cfg_path)
        self.hands_vae = self._load_single_vae(self.hands_cfg_path)
        self.motion_encoder = self._load_motion_encoder()
        self.dataloader = self._load_dataloader(self.dataset_cfg_path)

        print(">>> Start Building Motion Instance Graph...")
        count = 0
        
        for batch in tqdm(self.dataloader, desc="Processing"):
            if 'motion_upper' not in batch or 'motion_hands' not in batch: continue

            raw_upper = batch['motion_upper'].to(self.device).float()
            raw_hands = batch['motion_hands'].to(self.device).float()
            # motion = batch['motion'].to(self.device).float()
            # current_len = motion.shape[1]
            # if motion.shape[1] > 150: print(f"  - 长度大于150的样本: {current_len}") 

            # 元数据解析
            current_len = raw_upper.shape[1]
            file_id = f"Motion_Inst_{count:06d}"    #Motion_Inst_000000
            speaker_id = "unknown"
            
            # 注意: DataLoader 的 sample_idx 可能是 list 或 tensor
            if 'sample_name' in batch:
                s_idx = batch['sample_name']
                
                
                real_name = str(s_idx[0])
                safe_name = real_name.replace('/', '_').replace('\\', '_')  #2_scott_1_10_10_0
                emotion_tag = self._get_emotion_tag(safe_name)
                file_id = f"Motion_Inst_{safe_name}" #'Motion_Inst_2_scott_1_10_10_0'
                parts = safe_name.split('_')
                speaker_id = parts[0]   # 2

            # 1. 计算 VAE Latent
            in_upper = self._preprocess_motion(raw_upper)
            in_hands = self._preprocess_motion(raw_hands)
            with torch.no_grad():
                z_u, _ = self.upper_vae.encode_to_dist(in_upper)
                z_h, _ = self.hands_vae.encode_to_dist(in_hands)
                min_len = min(z_u.shape[1], z_h.shape[1])
                vae_latent_np = torch.cat([z_u[:, :min_len], z_h[:, :min_len]], dim=-1).cpu().numpy()

            # 2. 计算 CLIP Embedding
            clip_emb_np = np.zeros(512)
            if self.motion_encoder:
                with torch.no_grad():
                    # 尝试获取 263维 全身特征
                    full_body = None
                    full_body = batch['motion_h3d']
                    # if full_body.shape[0] != 150:
                    #     continue                    
                    
                    if full_body is not None:
                        full_body = full_body.to(self.device).float()
                        current_len = full_body.shape[1]
                        if full_body.shape[-1] == 623:
                            emb = self.motion_encoder.get_motion_embeddings(full_body, torch.tensor([current_len]))
                            clip_emb_np = emb.cpu().numpy().flatten()
                        else:
                            # 仅打印一次警告，防止刷屏
                            if count == 0: print(f"[Warn] Feature dim {full_body.shape[-1]} != 263. CLIP skipped.")
            # ==========================================
            # 🚀 [新增] 解析细粒度文本 (Word-level Text)
            # ==========================================
            # 🚀 [关键修改] 解析并存储单词级时间戳
            # ==========================================
            # 目标：把 [[[7.43, 7.88], 'okay'], ...] 存进节点属性
            slice_text = ""
            word_timings_json = "[]" # 默认空列表
            
            if 'text_segments' in batch:
                try:
                    # 1. 提取原始数据 (兼容 batch 维度)
                    raw_segments = batch['text_segments']
                    
                    # 处理可能的列表嵌套 (Batch wrapper)
                    if len(raw_segments) > 0 and isinstance(raw_segments[0], list) and isinstance(raw_segments[0][0], list):
                         segments = raw_segments[0] # 取出第一个样本的 segments
                    else:
                         segments = raw_segments

                    # 2. 格式化数据
                    # 我们希望存成: [{"word": "okay", "start": 7.43, "end": 7.88}, ...]
                    formatted_timings = []
                    word_list = []
                    
                    for item in segments:
                        # item 结构: [[start, end], 'word']
                        if len(item) >= 2:
                            time_range = item[0] # [7.43, 7.88]
                            word = str(item[1])  # 'okay'
                            
                            formatted_timings.append({
                                "word": word,
                                "start": float(time_range[0]),
                                "end": float(time_range[1])
                            })
                            word_list.append(word)
                    
                    # 3. 序列化 (图谱属性只能存字符串或数值，不能直接存对象)
                    slice_text = " ".join(word_list)
                    word_timings_json = json.dumps(formatted_timings)
                    
                except Exception as e:
                    print(f"[Warn] Failed to parse text_segments for {file_id}: {e}")
            # ==========================================
            # 🚀 [新增] 提取绝对时间 (Absolute Time)
            # ==========================================
            abs_start = 0.0
            if 'abs_start_time' in batch:
                val = batch['abs_start_time']
                # 处理 Tensor/List 封装
                if isinstance(val, torch.Tensor):
                    abs_start = val.item()
                elif isinstance(val, list):
                    abs_start = float(val[0])
                else:
                    abs_start = float(val)
            
            # 计算结束时间: Start + (Frames / FPS)
            duration_sec = current_len / self.fps
            abs_end = abs_start + duration_sec
            # ==========================================
            # ==========================================
            # 3. 保存节点 (存入 word_timings)
            # ==========================================
            save_name = f"{file_id}.npy"    #Motion_Inst_2_scott_1_10_10_0
            np.save(os.path.join(self.motion_assets_dir, save_name), {
                'vae_latent': vae_latent_np, 'duration': current_len
            })
            self.graph.add_node(
                file_id, 
                type="Motion_Instance",
                clip_embedding=json.dumps(clip_emb_np.tolist()),
                file_path=f"assets/{save_name}",
                speaker_id=speaker_id, 
                duration=int(current_len),
                # 🔥 [新增] 存入绝对时间戳！这是对齐的核心！
                start_time=float(abs_start),
                end_time=float(abs_end),
                
                # 🔥 新增：存入完整文本
                transcript=slice_text, 
                
                # 🔥🔥 新增：存入单词时间索引 (这就是那个 db_idx_2_gesture_labels 的替代品！)
                word_timings=word_timings_json,
                
                emotion_tag=emotion_tag, 
            )
            

            count += 1

        nx.write_gexf(self.graph, os.path.join(self.output_dir, "motion_instance_layer.gexf"))
        print(f"Done! Processed {count} instances.")
    def _get_emotion_tag(self, sample_name):
        """
        解析文件名获取情感标签 (根据用户指定：提取第4部分)
        Sample Name: 2_scott_0_1_1  -> 取 index[3] = 1 -> "neutral"
        Sample Name: 2_scott_0_73_1 -> 取 index[3] = 73 -> "anger"
        """
        try:
            parts = sample_name.split('_')
            
            # 🔥 修改点：提取第 4 部分 (Index 3)
            # 确保切分后的长度足够，避免越界
            if len(parts) > 3 and parts[3].isdigit():
                rec_id = int(parts[3])
            else:
                # 如果文件名格式不对 (例如只有 2_scott_0)，默认 neutral
                return "neutral" 

            # 映射逻辑 (保持不变)
            if 0 <= rec_id <= 64:
                return "neutral"
            elif 65 <= rec_id <= 72:
                return "happiness"
            elif 73 <= rec_id <= 80:
                return "anger"
            elif 81 <= rec_id <= 86:
                return "sadness"
            elif 87 <= rec_id <= 94:
                return "contempt"
            elif 95 <= rec_id <= 102:
                return "surprise"
            elif 103 <= rec_id <= 110:
                return "fear"
            elif 111 <= rec_id <= 118:
                return "disgust"
            else:
                return "neutral"
                
        except Exception as e:
            # print(f"[Warn] Failed to parse emotion from {sample_name}: {e}")
            return "neutral"
if __name__ == "__main__":
    import debugpy
    try:
        # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
        debugpy.listen(("localhost", 9503))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
    except Exception as e:
      pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--upper_cfg", type=str, required=True)
    parser.add_argument("--hands_cfg", type=str, required=True)
    parser.add_argument("--data_cfg", type=str, required=True)
    args = parser.parse_args()
    
    MotionInstanceBuilder(args.upper_cfg, args.hands_cfg, args.data_cfg).build()