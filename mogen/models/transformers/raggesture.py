from cv2 import norm
import torch
import torch.nn.functional as F
from torch import layer_norm, nn
import numpy as np
# import clip
import random
import math
from tqdm import tqdm
import time
import copy
import json
import os
import lmdb
import pyarrow
import pickle
import warnings
import shutil

from mogen.models.transformers.reward_adapter import StepAwareAdapter

from torch.nn.utils.rnn import pad_sequence

# ignore Future warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

from ..builder import SUBMODULES, build_attention
from .diffusion_transformer import DiffusionTransformer
from .rag.utils import TextFeatureExtractor, map_conns_to_prominence
from .rag.discourse_retrieval import discourse_retrieval
from .rag.llm_retrieval import llm_retrieval
from .rag.gesture_type_retrieval import gesture_type_retrieval
from ..utils.detr_utils import PositionEmbeddingLearned1D, PositionEmbeddingSine1D
from ..utils import rotation_conversions as rc
import sys
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
from models.temos.textencoder.distillbert_actor import DistilbertActorAgnosticEncoder

print("DEBUG: ✅ MotionCLIP Imported Successfully!")
HAS_TMR = True

# ================= 1. TMR Encoder Wrapper (新版) =================
class TMRMotionWrapper(nn.Module):
    """
    针对 TMR (Text-Motion-Retrieval) 动作编码器的封装
    """
    def __init__(self, model_dir, device):
        super().__init__()
        self.device = device
        print(f">>> [Loader] Loading TMR Motion Encoder from: {model_dir}")

        # 1. 实例化模型
        self.model = ActorAgnosticEncoder(nfeats=623, vae=True, num_layers=4)
        
        # 2. 加载权重
        ckpt_path = os.path.join(model_dir, "motion_epoch=299.ckpt") 
        if not os.path.exists(ckpt_path):
             import glob
             ckpts = glob.glob(os.path.join(model_dir, "motion_*.ckpt"))
             if ckpts: 
                 ckpt_path = ckpts[0]
        
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location=device)
            self.model.load_state_dict(state_dict)
            print(f"    ✅ Success: Weights loaded from {ckpt_path}")
        else:
            print(f"❌ [Error] Motion Ckpt not found in {model_dir}")

        # 【修改点 1】冻结模型参数
        # 作为推理时的“评价专家”，模型参数必须固定，避免主模型训练时对其产生干扰 
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(device)

        # 3. 均值方差加载 (用于归一化)
        self.register_buffer('mean', None)
        self.register_buffer('std', None)
        
        # ... (保持原有的 mean/std 加载逻辑不变) ...
        # (假设原逻辑已将 mean/std 赋值给 self.mean/self.std)

    def forward(self, raw_motion, lengths=None):
        with torch.no_grad():
            motions = raw_motion.to(self.device).float()
            
            if lengths is None:
                lengths = [motions.shape[1]] * motions.shape[0]

            if self.mean is not None and self.std is not None:
                motions = (motions - self.mean) / (self.std + 1e-8)
            
            dist = self.model(motions, lengths)
            
            # 【修改点 2】强制 L2 归一化
            # 归一化后的向量在点积时即代表余弦相似度，这是 ReAlign 进行对齐奖励计算的基础 
            return F.normalize(dist.loc, p=2, dim=-1)



class TMRTextWrapper(nn.Module):
    """
    针对 TMR (Text-Motion-Retrieval) 文本编码器的封装
    """
    def __init__(self, model_dir, device):
        super().__init__()
        self.device = device
        print(f">>> [Loader] Loading TMR Text Encoder from: {model_dir}")
        
        distilbert_path = '/Dataset4D/public/mas-liu.lianlian/code/SynTalker/ckpt/distilbert-base-uncased'
        if not os.path.exists(distilbert_path):
            distilbert_path = 'distilbert-base-uncased'

        # 1. 实例化模型
        self.textencoder = DistilbertActorAgnosticEncoder(distilbert_path, num_layers=4)
        
        # 2. 加载权重
        ckpt_path = os.path.join(model_dir, "text_epoch=299.ckpt") 
        if not os.path.exists(ckpt_path):
             import glob
             ckpts = glob.glob(os.path.join(model_dir, "text_*.ckpt"))
             if ckpts: 
                 ckpt_path = ckpts[0]
        
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location=device)
            self.textencoder.load_state_dict(state_dict)
            print(f"    ✅ Success: Weights loaded from {ckpt_path}")
        else:
            print(f"❌ [Error] Text Ckpt not found in {model_dir}")

        # 【修改点 1】进入评估模式并彻底冻结
        # 这能确保模型在主模型 load_state_dict 时即使有 Missing Keys 也不影响推理稳定性 
        self.textencoder.eval()
        for param in self.textencoder.parameters():
            param.requires_grad = False
        self.textencoder.to(device)

    def forward(self, text):
        if isinstance(text, str):
            text = [text]
            
        with torch.no_grad():
            dist = self.textencoder(text)
            
            # 【修改点 2】修复 Tensor 获取逻辑并进行归一化
            # 只有归一化后的语义特征才能准确用于计算文本-运动对齐奖励 
            text_feature = dist.loc 
            return F.normalize(text_feature, p=2, dim=-1)
        

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class FFN(nn.Module):

    def __init__(self, latent_dim, ffn_dim, dropout):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        y = x + y
        return y


class EncoderLayer(nn.Module):

    def __init__(self, sa_block_cfg=None, ca_block_cfg=None, ffn_cfg=None):
        super().__init__()
        self.sa_block = build_attention(sa_block_cfg)
        self.ffn = FFN(**ffn_cfg)

    def forward(self, **kwargs):
        if self.sa_block is not None:
            x = self.sa_block(**kwargs)
            kwargs.update({"x": x})
        if self.ffn is not None:
            x = self.ffn(**kwargs)
        return x
    

class LMDBDict:
    """
    A class to store the a dictionary as lmdb database file. As a new key is added
    the database is updated with the new key and value.
    for access, the key is used to retrieve the value from the lmdb database.
    """
    
    def __init__(self, db_path, torch_converter=False):
        self.db_path = db_path
        self.db = lmdb.open(
            db_path,
            map_size=int(1024 ** 3 * 300),
            readonly=False,
            lock=False,
        )
        self.torch_converter = torch_converter
    
    def __len__(self):
        with self.db.begin(write=False) as txn:
            return txn.stat()["entries"]

    def __setitem__(self, key, value):
        with self.db.begin(write=True) as txn:
            if self.torch_converter:
                if isinstance(value, torch.Tensor):
                    value = value.numpy()
                
                if isinstance(value, (list, tuple)):
                    value = [v.numpy() if isinstance(v, torch.Tensor) else v for v in value]

            # 将包含 numpy 数组的列表序列化为 bytes
            binary_data = pickle.dumps(value) 

            # 如果后续流程需要 buffer 对象
            v = memoryview(binary_data)
            txn.put(key.encode("ascii"), v)

    def __getitem__(self, key):
        with self.db.begin(write=False) as txn:
            value = txn.get(key.encode("ascii"))
            value = pickle.loads(value)

        if self.torch_converter:
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
                
            elif isinstance(value, list):
                value = [torch.from_numpy(v) if isinstance(v, np.ndarray) else v for v in value]
            
        return value
        

    def __del__(self):
        self.db.sync()
        self.db.close()

    def keys(self):
        with self.db.begin(write=False) as txn:
            return [key.decode("ascii") for key, _ in txn.cursor()]
    
    def values(self):
        raise NotImplementedError
    
    def items(self):
        for key in self.keys():
            yield key, self[key]

    def to_dict(self):
        return {k: v for k, v in self.items()}


class RetrievalDatabase(nn.Module):

    def __init__(
        self,
        dataset,
        motion_feat_dim=189,
        num_retrieval=None,
        topk=None,
        latent_dim=512,
        text_latent_dim=768,
        output_dim=512,
        num_layers=2,
        num_motion_layers=4,
        kinematic_coef=0.1,
        max_seq_len=150,
        motion_fps=15,
        motion_framechunksize=15,
        num_heads=8,
        ff_size=1024,
        stride=4,
        sa_block_cfg=None,
        ffn_cfg=None,
        dropout=0,
        lmdb_paths=None,
        new_lmdb_cache=False,
        stratified_db_creation=False,
        stratification_interval=15,
        kg_path=None,       # 图谱路径
        device=None,
        # retrieval_dict_path=None,
    ):
        super().__init__()
        self.num_retrieval = num_retrieval
        self.topk = topk
        self.latent_dim = latent_dim
        self.stride = stride
        self.kinematic_coef = kinematic_coef
        self.num_layers = num_layers
        self.num_motion_layers = num_motion_layers
        self.max_seq_len = max_seq_len

        self.retrieval_method = {
            "discourse": discourse_retrieval,
            "gesture_type": gesture_type_retrieval,
            "llm": llm_retrieval,
        }
        
        self.train_indexes = {}
        self.test_indexes = {}
        self.train_dbounds = {}
        self.test_dbounds = {}
        self.train_qbounds = {}
        self.test_qbounds = {}

        # breakpoint()

        self.dataset = dataset

        if new_lmdb_cache and os.path.exists(lmdb_paths):
            breakpoint()
            shutil.rmtree(lmdb_paths)
            os.makedirs(lmdb_paths)
        elif not os.path.exists(lmdb_paths):
            os.makedirs(lmdb_paths)
        
        

        
        self.idx_2_text = LMDBDict(os.path.join(lmdb_paths, "idx_2_text"), torch_converter=True)
        self.idx_2_sense = LMDBDict(os.path.join(lmdb_paths, "idx_2_sense"))
        self.idx_2_discbounds = LMDBDict(os.path.join(lmdb_paths, "idx_2_discbounds"))
        self.idx_2_gesture_labels = LMDBDict(os.path.join(lmdb_paths, "idx_2_gesture_labels"))
        self.idx_2_prominence = LMDBDict(os.path.join(lmdb_paths, "idx_2_prominence"))
        self.idx_2_gestprom = LMDBDict(os.path.join(lmdb_paths, "idx_2_gestprom"))
        self.dataset = dataset

        # breakpoint()
        if new_lmdb_cache:
            print("Creating retrieval databases")
            for smp_idx, smp in tqdm(enumerate(dataset)):
                # do random selection from self.idx_2_sense, self.idx_2_discbounds, self.idx_2_prominence, self.idx_2_text, idx_2_gesture_labels
                # sample every 30th/15th example from one sample sequence
                # verify that those sampled actually amount upto the length of the sequence
                # breakpoint()
                if stratified_db_creation:
                    per_sample_idx = smp["sample_name"].split("/")[1]
                    if int(per_sample_idx) % stratification_interval != 0:
                        if smp_idx == len(dataset) - 1: break
                        continue

                # breakpoint()  # check the dataset
                speaker_id = int(smp["speaker_id"][0].item())
                # breakpoint()
                if new_lmdb_cache or not os.path.exists(os.path.join(lmdb_paths, "idx_2_text")):
                    self.idx_2_text[smp["sample_name"]] = (smp["text_feature"], speaker_id)

                if new_lmdb_cache or not os.path.exists(os.path.join(lmdb_paths, "idx_2_sense")):
                    self.idx_2_sense[smp["sample_name"]] = [speaker_id] + [(d[1], d[0]) for d in smp["discourse"]]  # speaker_id, sense, text
                
                if new_lmdb_cache or not os.path.exists(os.path.join(lmdb_paths, "idx_2_discbounds")):
                    self.idx_2_discbounds[smp["sample_name"]] = [(d[1], d[0], d[4], d[5], d[6], d[7]) for d in smp["discourse"]]  # sense, text, disco_start, disco_end, conn_start, conn_end

                if new_lmdb_cache or not os.path.exists(os.path.join(lmdb_paths, "idx_2_gesture_labels")):
                    self.idx_2_gesture_labels[smp["sample_name"]] = [speaker_id] + smp["gesture_labels"]
                

                # filter out the relevant prominance values according to
                # the connectives in disco conns
                # breakpoint()  # check following code
                if new_lmdb_cache or not os.path.exists(os.path.join(lmdb_paths, "idx_2_prominence")):
                    smp_conns = []
                    for disco in smp["discourse"]:
                        smp_conns.append(disco[0])
                    relevant_dps = map_conns_to_prominence(smp_conns, smp["prominence"])

                    # if len(relevant_dps) > 1 and relevant_dps[0] is not None:
                    #     if "." in relevant_dps[0][0]:
                    #         breakpoint()

                    self.idx_2_prominence[smp["sample_name"]] = relevant_dps

                if new_lmdb_cache or not os.path.exists(os.path.join(lmdb_paths, "idx_2_gestprom")):
                    smp_gest_words = [s["word"] for s in smp["gesture_labels"]]
                    relevant_gest_dps = map_conns_to_prominence(smp_gest_words, smp["prominence"])
                    self.idx_2_gestprom[smp["sample_name"]] = relevant_gest_dps

                if smp_idx == len(dataset) - 1:
                    
                    break

            print("Retrival databases creation finished")

        # TODO: stratify existing databases 

        # load the databases into memory as dictionaries
        self.idx_2_text = self.idx_2_text.to_dict()
        self.idx_2_sense = self.idx_2_sense.to_dict()
        self.idx_2_discbounds = self.idx_2_discbounds.to_dict()
        self.idx_2_gesture_labels = self.idx_2_gesture_labels.to_dict()
        self.idx_2_prominence = self.idx_2_prominence.to_dict()
        self.idx_2_gestprom = self.idx_2_gestprom.to_dict()
        

        
        self.feature_cache_tensor = pad_sequence([f[0] for f in self.idx_2_text.values()], batch_first=True)
        self.sample_names = {i: s for i, s in enumerate(self.idx_2_text.keys())}



        # breakpoint()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.text_latent_dim = text_latent_dim

        self.motion_fps = motion_fps
        self.motion_framechunksize = motion_framechunksize

        self.device=device  



        # 定义缓存文件路径
        name_to_idx_path = os.path.join(lmdb_paths, "name_to_idx.json")

        # 逻辑：如果要求强制刷新，或者缓存文件不存在，则重新构建
        if new_lmdb_cache or not os.path.exists(name_to_idx_path):
            print("[Init] Building Global Dataset Index Mapping (This may take a while)...")
            self.name_to_idx = {}
            
            for original_idx in tqdm(range(len(self.dataset)), desc="Indexing Dataset"):
                smp_name = self.dataset[original_idx]["sample_name"]
                self.name_to_idx[smp_name] = original_idx 

            # 保存到本地 JSON 文件
            import json
            with open(name_to_idx_path, 'w', encoding='utf-8') as f:
                json.dump(self.name_to_idx, f)
            print(f"[Init] ✅ Saved index mapping to {name_to_idx_path}")

        else:
            # 逻辑：如果文件存在且不需要强制刷新，直接极速加载
            import json
            print(f"[Init] Loading Global Dataset Index Mapping from cache...")
            with open(name_to_idx_path, 'r', encoding='utf-8') as f:
                self.name_to_idx = json.load(f)
            print(f"[Init] ✅ Fast loaded {len(self.name_to_idx)} entries from cache.")
        
        # [新增] 构建反向映射 (ID -> Name) 用于 Neighbor Proxy 校验
        print("[Init] Building Inverse Index (ID -> Name)...")
        self.idx_2_name = {v: k for k, v in self.name_to_idx.items()}

        # [新增] 加载静态知识图谱
        import networkx as nx
        self.kg_graph = None
        if kg_path and os.path.exists(kg_path):
            print(f"[KG] Loading Static Graph from {kg_path}...")
            self.kg_graph = nx.read_gexf(kg_path)
            # 建立 name_to_idx 映射，方便从 Graph ID 找到 Dataset Index
            
            self._build_vector_cache()
        
        tmr_motion_dir="/Dataset4D/public/mas-liu.lianlian/pretrained_models/beatx_1-30_amass_h3d_tmr/"
        tmr_text_dir="/Dataset4D/public/mas-liu.lianlian/pretrained_models/beatx_1-30_amass_h3d_tmr/"

        self.motion_encoder = self._load_motion_encoder(tmr_motion_dir)
        self.text_encoder = self._load_text_encoder(tmr_text_dir)
    
    def _build_vector_cache(self):
        """
        [新增] 预计算 Path B 所需的向量缓存
        将图谱中的 clip_embedding 提取为 Tensor，避免推理时实时解析 JSON
        """
        print("[Cache] Building Motion Vector Cache from Graph...")
        
        cache_vecs = []
        cache_idxs = [] # 存储对应的 dataset int index
        
        count = 0
        for node, attrs in self.kg_graph.nodes(data=True):
            # 筛选动作节点且有 embedding
            if node.startswith("Motion_Inst") and "clip_embedding" in attrs:
                try:
                    # 1. 解析 Sample Name -> Index
                    # 假设 ID 格式: Motion_Inst_{sample_name} 或包含后缀
                    # 这里直接用简单的 replace，或者复用 parse_smp_name_from_id 逻辑
                    smp_name = node.replace("Motion_Inst_", "")
                    # 如果有后缀导致匹配失败，可能需要 split('_') 等更复杂的逻辑
                    # 这里假设 name_to_idx 能匹配上
                    parts = smp_name.split("_")
                    if len(parts) > 1:
                        smp_name = "_".join(parts[:-1]) + "/" + parts[-1]
                    
                    if smp_name in self.name_to_idx:
                        dataset_idx = self.name_to_idx[smp_name]
                        
                        
                        # 2. 解析 Embedding
                        emb_str = attrs["clip_embedding"]
                        # print(emb_str.shape)
                        vec = json.loads(emb_str) if isinstance(emb_str, str) else emb_str
                        
                        cache_vecs.append(vec)
                        cache_idxs.append(dataset_idx)
                        count += 1
                except Exception as e:
                    continue
        
        if count > 0:
            # 转为 Tensor
            tensor_vecs = torch.tensor(cache_vecs, dtype=torch.float32)
            tensor_idxs = torch.tensor(cache_idxs, dtype=torch.long)
            
            # [关键] 预先做归一化 (L2 Normalize)，推理时直接点积即可
            tensor_vecs = F.normalize(tensor_vecs, p=2, dim=1)
            
            # [关键] 使用 register_buffer 注册为模型的一部分
            # 这样执行 model.to('cuda') 时，缓存会自动转到 GPU
            self.register_buffer("cached_motion_embeddings", tensor_vecs)
            self.register_buffer("cached_motion_indices", tensor_idxs)
            
            print(f"[Cache] ✅ Successfully cached {count} motion vectors. Shape: {tensor_vecs.shape}")
        else:
            print("[Cache] ⚠️ No motion embeddings found in graph!")
            self.register_buffer("cached_motion_embeddings", None)
            self.register_buffer("cached_motion_indices", None)

    def _load_motion_encoder(self, model_dir):
        """
        在内部直接完成声明、加载、冻结三部曲
        """
        if not os.path.exists(model_dir):
            print(f"⚠️ [Warn] Motion Encoder path not found, skipping: {model_dir}")
            return None

        try:
            # 这里的 TMRMotionWrapper 内部已经写了 torch.load(ckpt_path)
            encoder = TMRMotionWrapper(model_dir, self.device)
            
            # 显式冻结，防止主模型训练时意外修改这些“专家模型”的参数
            encoder.eval()
            for param in encoder.parameters():
                param.requires_grad = False
            
            print("✅ [RetrievalDatabase] Motion Encoder self-loaded and frozen.")
            return encoder
        except Exception as e:
            print(f"❌ [Error] Failed to self-load Motion Encoder: {e}")
            return None
    def _load_text_encoder(self, model_dir):
        """
        同理，自洽加载文本编码器
        """
        if not os.path.exists(model_dir):
            return None
        try:
            encoder = TMRTextWrapper(model_dir, self.device)
            encoder.eval()
            for param in encoder.parameters():
                param.requires_grad = False
            print("✅ [RetrievalDatabase] Text Encoder self-loaded and frozen.")
            return encoder
        except Exception as e:
            print(f"❌ [Error] Failed to self-load Text Encoder: {e}")
            return None
    def find_best_window_match(self, full_latents, text_emb, fps=15, frame_chunk_size=1):
        """
        [新增] 多尺度滑动窗口搜索 (GPU加速版)
        在 Latent 空间寻找与 Text Embedding 最相似的片段。
        """
        # 1. 维度检查与归一化
        # text_emb: [D] or [1, D] -> [D]
        if text_emb.dim() == 2: text_emb = text_emb.squeeze(0)
        
        # 归一化是计算余弦相似度的前提
        text_emb = F.normalize(text_emb, p=2, dim=0)
        full_latents = F.normalize(full_latents, p=2, dim=1) # [T, D]

        # 2. 定义搜索尺度 (秒) -> Latent步数
        # 假设 1个 latent = frame_chunk_size 帧 (比如 4)
        latent_fps = fps / frame_chunk_size
        scales_sec = [2.0, 3.0, 4.0] # 搜索 2s, 3s, 4s 的窗口
        
        best_score = -999.0
        # 默认返回整段
        total_duration = full_latents.shape[0] / latent_fps
        best_window = (0.0, total_duration)

        for duration in scales_sec:
            win_len_lat = int(duration * latent_fps)
            stride_lat = max(1, int(0.5 * latent_fps)) # 0.5s 步长
            
            # 如果动作比窗口还短，跳过
            if win_len_lat >= full_latents.shape[0]:
                continue
            
            try:
                # 3. 快速滑动窗口 (Unfold)
                # Input: [T, D] -> Unfold -> [Num_Windows, D, Win_Len]
                # dim 0 is Time, we unfold on it.
                windows = full_latents.unfold(0, win_len_lat, stride_lat) 
                
                # Mean Pooling: 计算每个窗口内的平均特征向量
                # [Num_Windows, D, Win_Len] -> mean(dim=2) -> [Num_Windows, D]
                win_feats = windows.mean(dim=2)
                win_feats = F.normalize(win_feats, p=2, dim=1)
                
                # 4. 批量计算相似度
                # [Num_Windows, D] @ [D] -> [Num_Windows]
                scores = torch.matmul(win_feats, text_emb)
                
                # 5. 找最大值
                curr_best_val, curr_best_idx = torch.max(scores, dim=0)
                
                if curr_best_val > best_score:
                    best_score = curr_best_val.item()
                    
                    # 转换回秒
                    start_lat_idx = curr_best_idx.item() * stride_lat
                    end_lat_idx = start_lat_idx + win_len_lat
                    
                    best_window = (
                        start_lat_idx / latent_fps, 
                        end_lat_idx / latent_fps
                    )
            except Exception as e:
                # print(f"Window search warning: {e}")
                continue
                
        return best_window
    def generate_src_mask(self, T, length):
        B = len(length)
        src_mask = torch.ones(B, T)
        for i in range(B):
            for j in range(length[i], T):
                src_mask[i, j] = 0
        return src_mask

    def retrieve(
        self,
        retr_method,
        text,
        text_features,
        audio,
        discourse,
        gesture_labels,
        text_times,
        prominence,
        speaker_id,
        llm_full_context,
        llm_focus_window,
        # length,
        idx=None,
        device = 'cpu',

    ):
        # train cache is in form of two dicts: self.train_indexes and self.train_dbounds
        # test cache is in form of two dicts: self.test_indexes and self.test_dbounds

        # self.train_indexes is a dict of query idx to dict of retrival types to list of smp indexes
        # self.train_dbounds is a dict of query idx to dict of retrival types to dict of smp indexes to bounds
        # breakpoint()
        assert retr_method in ["gesture_type", "discourse", "llm"] # "llm"
        # print(f"Retrieval method: {retr_method}")
        if self.training and idx in self.train_indexes and idx is not None:
            # idx = idx.item()
            # breakpoint()
            multiple_db_indexes = self.train_indexes[idx]
            multiple_db_bounds = self.train_dbounds[idx]
            multiple_query_bounds = self.train_qbounds[idx]

            # select the db_indexes and db_bounds randomly during training
            per_idx_retrmethods = list(multiple_db_indexes.keys())
            if len(per_idx_retrmethods) == 0:
                return {}, {}, {}
            train_retr_method = random.choice(per_idx_retrmethods)

            db_indexes = multiple_db_indexes[train_retr_method]
            db_bounds = multiple_db_bounds[train_retr_method]
            query_bounds = multiple_query_bounds[train_retr_method]

            data = {}
            # bounds = {}
            for query_w_idx, smp_idxs in db_indexes.items():
                data[query_w_idx] = [s_i for s_i in smp_idxs if s_i != idx]
                data[query_w_idx] = data[query_w_idx][: self.topk]
                random.shuffle(data[query_w_idx])
                data[query_w_idx] = data[query_w_idx][: self.num_retrieval]

            return data, db_bounds, query_bounds

        elif not self.training and idx in self.test_indexes and idx is not None:
            
            multiple_db_indexes = self.test_indexes[idx]
            multiple_db_bounds = self.test_indexes[idx]
            multiple_query_bounds = self.test_qbounds[idx]

            if retr_method not in multiple_db_indexes:
                print(
                    f"WARNUNG: Retrieval method {retr_method} not found for idx {idx}"
                )
                return {}, {}, {}

            # select the db_indexes and db_bounds based on the retr_method during testing
            db_indexes = multiple_db_indexes[retr_method]
            db_bounds = multiple_db_bounds[retr_method]
            query_bounds = multiple_query_bounds[retr_method]

            if len(db_indexes) == 0:
                print(
                    f"WARNUNG: No samples found for idx {idx} for retr_method {retr_method}"
                )
                # return {}, {}, {}

            data = {}
            # bounds = {}
            for query_w_idx, smp_idxs in db_indexes.items():
                data[query_w_idx] = [s_i for s_i in smp_idxs if s_i != idx]
                # data[query_w_idx] = data[query_w_idx][: self.topk]
                # random.shuffle(data[query_w_idx])
                data[query_w_idx] = data[query_w_idx][: self.num_retrieval]

            return data, db_bounds, query_bounds
        else:
            # base_method_args =
            # breakpoint() 
            method_args = {}
            for retr_m in self.retrieval_method:

                
                method_args[retr_m] = {
                    "text": text,
                    "speaker_id": speaker_id,
                    "encoded_text": text_features,
                    "text_feat_cache": self.idx_2_text,
                }
                if retr_m == "gesture_type":
                    method_args[retr_m]["gesture_labels"] = gesture_labels
                    method_args[retr_m][
                        "db_idx_2_gesture_labels"
                    ] = self.idx_2_gesture_labels
                elif retr_m == "llm":
                    method_args[retr_m]["text_times"] = text_times
                    method_args[retr_m][
                        "db_idx_2_gesture_labels"
                    ] = self.idx_2_gesture_labels
                    method_args[retr_m]["prominence"] = prominence
                    method_args[retr_m]["db_idx_2_prominence"] = self.idx_2_gestprom
                    # ================= [在这里添加] =================
                    # 传入图谱组件以支持 Path A (图遍历) 和 Path B (向量对齐)
                    method_args[retr_m]["kg_graph"] = getattr(self, 'kg_graph', None)
                    method_args[retr_m]["name_to_idx"] = getattr(self, 'name_to_idx', {})
                    # 传入编码器用于 Path B (确保 self.text_encoder 已定义或通过参数传入)
                    method_args[retr_m]["text_model"] = getattr(self, 'text_encoder', None) 
                    method_args[retr_m]["device"] = device # 确保 device 可用
                    # 注意：如果 graph 加载失败，getattr 返回 None
                    method_args[retr_m]["cached_embeds"] = getattr(self, 'cached_motion_embeddings', None)
                    method_args[retr_m]["cached_idxs"] = getattr(self, 'cached_motion_indices', None)

                    method_args[retr_m]["full_text"] = llm_full_context
                    method_args[retr_m]["focus_window"] = llm_focus_window
                    method_args[retr_m]["idx_2_name"] = getattr(self, 'idx_2_name', None)
                    method_args[retr_m]["motion_model"] = getattr(self, 'motion_encoder', None)
                    method_args[retr_m]["dataset_handle"] = getattr(self, 'dataset', None)
                    # ===============================================

                elif retr_m == "discourse":
                    method_args[retr_m]["discourse"] = discourse
                    method_args[retr_m]["prominence"] = prominence
                    method_args[retr_m]["db_idx_2_sense"] = self.idx_2_sense
                    method_args[retr_m]["db_idx_2_discbounds"] = self.idx_2_discbounds
                    method_args[retr_m]["db_idx_2_prominence"] = self.idx_2_prominence
                elif retr_m == "prosody":
                    method_args[retr_m]["audio"] = audio
                    method_args[retr_m]["prominence"] = prominence
                    method_args[retr_m]["db_idx_2_prominence"] = self.idx_2_prominence
                    raise NotImplementedError
                
                else:
                    raise NotImplementedError

            self.train_indexes[idx] = {}
            self.train_dbounds[idx] = {}
            self.train_qbounds[idx] = {}
            self.test_indexes[idx] = {}
            self.test_dbounds[idx] = {}
            self.test_qbounds[idx] = {}

            if not self.training:
                # use retrieval method to get the sample indexes and bounds
                # instead of randomly selecting one
                sample_indexes, sample_bounds, query_bounds = self.retrieval_method[
                    retr_method
                ](**method_args[retr_method])

                self.test_indexes[idx].update({retr_method: sample_indexes})
                self.test_dbounds[idx].update({retr_method: sample_bounds})
                self.test_qbounds[idx].update({retr_method: query_bounds})

            else:
                raise NotImplementedError("Not released for training for retrieval")

                per_idx_retrmethods = list(self.train_indexes[idx].keys())
                if len(per_idx_retrmethods) == 0:
                    return {}, {}, {}
                
                
                train_retr_method = random.choice(per_idx_retrmethods)
                # print(f"-----Selected retrieval method: {train_retr_method}")

                sample_indexes = self.train_indexes[idx][train_retr_method]
                sample_bounds = self.train_dbounds[idx][train_retr_method]
                query_bounds = self.train_qbounds[idx][train_retr_method]

                # breakpoint()

            data = {}
            # bounds = {}
            for query_w_idx, smp_idxs in sample_indexes.items():
                data[query_w_idx] = [s_i for s_i in smp_idxs if s_i != idx]

                data[query_w_idx] = data[query_w_idx][: self.num_retrieval]

            return data, sample_bounds, query_bounds

    def forward(
        self, conditions, lengths, device, idx=None, llm_full_context = None,llm_focus_window = None,retrieval_method="gesture_type", gesture_rep_encoder=None
    ):
        B = len(conditions["text"])
        all_indexes = []
        all_masked_motions = []
        raw_masked_motions = []
        raw_masked_motions_aa = []
        raw_masked_trans = []
        raw_masked_facial = []
        all_words = []
        all_raw_words = []
        all_type2words = []

        all_retr_startends = []
        all_query_startends = []
        all_retr_latents = []
        start = time.time()
        # timess = []
        for b_ix in range(B):
            retr_indexes, retr_bounds, query_bounds = self.retrieve(
                retrieval_method,
                text=conditions["text"][b_ix],
                text_features=conditions["text_features"][b_ix],
                audio=conditions["audio"][b_ix],
                discourse=conditions["discourse"][b_ix],
                gesture_labels=conditions["gesture_labels"][b_ix],
                text_times=conditions["text_times"][b_ix],
                prominence=conditions["prominence"][b_ix],
                speaker_id=conditions["speaker_ids"][b_ix, 0].item(),
                # lengths[b_ix],
                idx=idx[b_ix] if idx is not None else None,
                device=device,
                llm_full_context=llm_full_context,
                llm_focus_window=llm_focus_window,
            )
            all_indexes.append(retr_indexes)
            
            

            batch_masked_motions = []
            batch_words = []
            batch_type2words = {}

            
            
            zero_motion = torch.zeros((self.max_seq_len // self.motion_framechunksize * 4 + 3, self.latent_dim)).to(device)

            zero_text = torch.zeros((self.max_seq_len // self.motion_framechunksize * 4 + 3, self.text_latent_dim)).to(device)

            zero_raw_motion = torch.zeros_like(self.dataset[0]["motion"]).to(device)
            zero_raw_text = torch.zeros_like(self.dataset[0]["word"]).to(device)
            zero_raw_motion_aa = torch.zeros_like(self.dataset[0]["motion"]).to(device)
            
            zero_raw_trans = torch.zeros_like(self.dataset[0]["trans"]).to(device)
            zero_raw_facial = torch.zeros_like(self.dataset[0]["facial"]).to(device)

            text_encoded = conditions["text_enc"][b_ix]
            # breakpoint() 
            prev_end_frame = -1

            retr_startend = {}
            query_startend = {}
            retrlatents_uncropped = {}

            # breakpoint()
            for query_point_idx, smp_idxs in retr_indexes.items():
                if len(smp_idxs) == 0:
                    continue

                if query_point_idx not in query_bounds:
                    continue

                query_bound = query_bounds[query_point_idx]
                query_word, query_type, query_start, query_end = query_bound

                if query_start > query_end:
                    continue

                assert len(smp_idxs) == self.num_retrieval == 1
                for smp_idx in smp_idxs:
                    
                    retr_motion = self.dataset[smp_idx]["motion"].unsqueeze(0).to(device)
                    retr_motion_upper = (self.dataset[smp_idx]["motion_upper"]).unsqueeze(0).to(device)
                    retr_motion_lower = (self.dataset[smp_idx]["motion_lower"]).unsqueeze(0).to(device)
                    retr_motion_face = (self.dataset[smp_idx]["motion_face"]).unsqueeze(0).to(device)
                    retr_motion_facial = (self.dataset[smp_idx]["facial"]).unsqueeze(0).to(device)
                    retr_motion_hands = (self.dataset[smp_idx]["motion_hands"]).unsqueeze(0).to(device)
                    retr_motion_transl = (self.dataset[smp_idx]["trans"]).unsqueeze(0).to(device)
                    retr_motion_contact = (self.dataset[smp_idx]["contact"]).unsqueeze(0).to(device)
                    retr_motion_mask = (self.dataset[smp_idx]["motion_mask"]).unsqueeze(0).to(device)

                    retr_text = self.dataset[smp_idx]["word"].unsqueeze(0).to(device)
                    retr_audio = self.dataset[smp_idx]["audio"].unsqueeze(0).to(device)
                    retr_spkid = self.dataset[smp_idx]["speaker_id"].unsqueeze(0).to(device)
                    retr_motion_aa = retr_motion.clone()
                    

                    
                    if retr_motion.shape[0] == 0:
                        continue

                    assert gesture_rep_encoder is not None
                    
                    retr_motion_latent, re_lat_motion_mask = gesture_rep_encoder.encode(
                        retr_motion_upper, retr_motion_lower, retr_motion_face, retr_motion_hands, retr_motion_transl, retr_motion_facial, retr_motion_contact, retr_motion_mask
                    )
                    

                    retr_motion_latent = retr_motion_latent.squeeze(0)
                    retr_motion = retr_motion.squeeze(0) # .detach()
                    retr_motion_aa = retr_motion_aa.squeeze(0) #.detach()
                    retr_motion_transl = retr_motion_transl.squeeze(0) #.detach()
                    retr_motion_facial = retr_motion_facial.squeeze(0) #.detach()

                    motion_len = self.max_seq_len #zero_motion.shape[0] # change this according to zero_motion # check axis of zero_motion

                    
                    # 2. 获取原始边界信息
                    retr_word, retr_type, retr_start, retr_end = retr_bounds[query_point_idx][smp_idx]

                    
                    
                    # logging for testing
                    batch_type2words[query_point_idx] = (
                        query_word,
                        query_type,
                        retr_word,
                        retr_type,
                    )

                    # motion features Rm 查询时间边界检查
                    query_start = max(0, query_start)
                    query_end = min(motion_len / self.motion_fps, query_end)

                    # 转换时间单位为帧级别
                    query_start = int(query_start * self.motion_fps)
                    query_end = int(query_end * self.motion_fps)

                    # if query_start == query_end:
                    #     continue
                    
                    #计算潜在空间（latent space）中的起始和结束索引
                    query_lat_start = query_start // self.motion_framechunksize
                    query_lat_end = query_end // self.motion_framechunksize + 1
                    if query_lat_start >= query_lat_end:
                        breakpoint() # check wth happened here
                    
                    # 0.6 before and 0.3 sec after the stroke. 
                    # currently assuming stroke in the middle. 
                    # time reference frame
                    # breakpoint()
                    # if (retrieval_method == "gesture_type" or retrieval_method == "llm") \
                    #     and (retr_end - retr_start) > 0.9:
                    #     # reduced padding for gesture retrieval 
                    #     # because of large annotation duration in BEAT dataset
                    #     retr_start = max(0, retr_start - 0.2)
                    #     retr_end = min(motion_len / self.motion_fps, retr_end + 0.1)
                    # else:
                    #     retr_start = max(0, retr_start - 0.666)  # half second padding 
                    #     retr_end = min(motion_len / self.motion_fps, retr_end + 0.333)  # half second padding
                    # padding also affects how much of an overlap you have with
                    # other retr motions

                    if (retr_type not in ["intent_node", "graph_match"]) and (retr_end <= 0.0):
                        # 获取动作真实时长
                        full_len = retr_motion.shape[1] / self.motion_fps
                        
                        # 策略：居中截取 3秒 (防止动作过长覆盖其他内容)
                        target_duration = min(full_len, 3.0)
                        center = full_len / 2
                        retr_start = max(0, center - target_duration / 2)
                        retr_end = min(full_len, center + target_duration / 2)

                    # =========================================================
                    # 步骤 B: Padding 逻辑 (唯一入口)
                    # =========================================================
                    
                    # 分支 1: 图谱精确匹配 (Intent Node)
                    # 只有这种情况需要 Padding，因为原始数据是紧凑的
                    if retr_type == "intent_node" or retr_type == "graph_match":
                        if (retr_end - retr_start) > 0.9:
                            # 长动作：少量扩充
                            retr_start = max(0, retr_start - 0.2)
                            retr_end = min(motion_len / self.motion_fps, retr_end + 0.1)
                        else:
                            # 短动作：多扩充一点上下文
                            retr_start = max(0, retr_start - 0.666)  
                            retr_end = min(motion_len / self.motion_fps, retr_end + 0.333)
                    
                    # 分支 2: 向量语义匹配 (Vector Fallback)
                    # 我们已经在 llm_retrieval 里通过滑动窗口找好了最佳片段，或者在步骤 A 做了截取
                    # 这里【绝对不要】再加 Padding，否则会破坏语义对齐
                    else:
                        retr_start = max(0, retr_start)
                        retr_end = min(motion_len / self.motion_fps, retr_end)

                    # frame reference frame
                    retr_start = int(retr_start * self.motion_fps)
                    retr_end = int(retr_end * self.motion_fps)

                    # breakpoint()
                    if retr_start == retr_end:
                        continue
                    
                    if retr_end == motion_len:
                        retr_end = motion_len - 1
                        retr_start = max(0, retr_start - 1)

                    retr_lat_start = retr_start // self.motion_framechunksize
                    retr_lat_end = retr_end // self.motion_framechunksize + 1
                    if retr_lat_start >= retr_lat_end:
                        breakpoint() # check wtf happened here

                    query_mid = (query_start + query_end) // 2
                    query_mid_lat = query_mid // self.motion_framechunksize

                    latent_len = (zero_motion.shape[0] - 3) // 4
                    assert latent_len == motion_len // self.motion_framechunksize

                    # breakpoint()

                    
                    retr_window_lat_u = retr_motion_latent[retr_lat_start:retr_lat_end]
                    retr_window_lat_h = retr_motion_latent[latent_len + 1 + retr_lat_start: latent_len + 1 + retr_lat_end]
                    retr_window_lat_f = retr_motion_latent[2 * latent_len + 2 + retr_lat_start: 2 * latent_len + 2 + retr_lat_end]
                    retr_window_lat_lt = retr_motion_latent[3 * latent_len + 3 + retr_lat_start: 3 * latent_len + 3 + retr_lat_end]

                    retr_motion_raw = retr_motion[retr_lat_start*self.motion_framechunksize:retr_lat_end*self.motion_framechunksize]
                    retr_motion_raw_aa = retr_motion_aa[retr_lat_start*self.motion_framechunksize:retr_lat_end*self.motion_framechunksize]
                    retr_trans_raw = retr_motion_transl[retr_lat_start*self.motion_framechunksize:retr_lat_end*self.motion_framechunksize]
                    retr_facial_raw = retr_motion_facial[retr_lat_start*self.motion_framechunksize:retr_lat_end*self.motion_framechunksize]
                    retr_length_raw = retr_motion_raw.shape[0]

                    # breakpoint()
                    retr_length_lat = retr_window_lat_u.shape[0]
                    assert retr_length_lat > 0
                    if retr_length_lat == 1:
                        side_length = retr_length_lat // 2
                        start_lat = query_mid_lat - side_length
                        end_lat = query_mid_lat + side_length + 1
                    elif retr_length_lat == 2:
                        side_length = retr_length_lat // 2
                        start_lat = query_mid_lat
                        end_lat = query_mid_lat + side_length + 1
                    elif retr_length_lat % 2 == 1:
                        side_length = retr_length_lat // 2
                        start_lat = query_mid_lat - side_length - 1
                        end_lat = query_mid_lat + side_length
                    else:
                        side_length = retr_length_lat // 2
                        start_lat = query_mid_lat - side_length
                        end_lat = query_mid_lat + side_length

                    if start_lat < 0:
                        start_lat = 0
                        end_lat = retr_length_lat
                    
                    if end_lat > latent_len:
                        start_lat -= end_lat - latent_len
                        end_lat = latent_len

                    if start_lat < prev_end_frame:
                        start_lat = prev_end_frame
                        end_lat = start_lat + retr_length_lat
                        if end_lat > latent_len:
                            end_lat = latent_len
                            retr_length_lat = end_lat - start_lat
                            # breakpoint() # shouldnt it be retr_window_lat[start_lat:end_lat]?
                            if retr_length_lat <= 0:
                                continue
                            retr_window_lat_u = retr_window_lat_u[:retr_length_lat]
                            # retr_window_lat_l = retr_window_lat_l[:retr_length_lat]
                            
                            retr_window_lat_h = retr_window_lat_h[:retr_length_lat]
                            retr_window_lat_f = retr_window_lat_f[:retr_length_lat]
                            retr_window_lat_lt = retr_window_lat_lt[:retr_length_lat]
                            # retr_window_lat_t = retr_window_lat_t[:retr_length_lat]


                            retr_motion_raw = retr_motion_raw[:retr_length_lat*self.motion_framechunksize]
                            retr_motion_raw_aa = retr_motion_raw_aa[:retr_length_lat*self.motion_framechunksize]
                            retr_trans_raw = retr_trans_raw[:retr_length_lat*self.motion_framechunksize]
                            retr_facial_raw = retr_facial_raw[:retr_length_lat*self.motion_framechunksize]

                            # update retr_lat_end
                            retr_lat_end = retr_lat_start + retr_length_lat


                            # breakpoint()
                            assert retr_window_lat_u.shape[0] == retr_length_lat

                    prev_end_frame = end_lat

                    
                    # append retr_latents to the list of retr_latents
                    retrlatents_uncropped[query_point_idx] = {
                        "retr_motion_latent": retr_motion_latent.unsqueeze(0), # 1, T, D
                        "retr_text": retr_text,
                        "retr_audio": retr_audio,
                        "retr_spkid": retr_spkid,
                        "retr_motion_mask": re_lat_motion_mask,
                    }
                    # append the retr_start and retr_end to the list of retr_startend list
                    retr_startend[query_point_idx] = (retr_lat_start, retr_lat_end)
                    # append the query_start and query_end to the list of query_startend list
                    query_startend[query_point_idx] = (start_lat, end_lat)

                    
                    zero_motion[start_lat:end_lat] = retr_window_lat_u
                    zero_motion[latent_len + 1 + start_lat: latent_len + 1 + end_lat] = retr_window_lat_h
                    zero_motion[2 * latent_len + 2 + start_lat: 2 * latent_len + 2 + end_lat] = retr_window_lat_f
                    zero_motion[3 * latent_len + 3 + start_lat: 3 * latent_len + 3 + end_lat] = retr_window_lat_lt
                    

                    zero_raw_motion[start_lat*self.motion_framechunksize:end_lat*self.motion_framechunksize] = retr_motion_raw
                    zero_raw_motion_aa[start_lat*self.motion_framechunksize:end_lat*self.motion_framechunksize] = retr_motion_raw_aa
                    zero_raw_trans[start_lat*self.motion_framechunksize:end_lat*self.motion_framechunksize] = retr_trans_raw
                    zero_raw_facial[start_lat*self.motion_framechunksize:end_lat*self.motion_framechunksize] = retr_facial_raw
                    
                    
                    if query_start >= query_end:
                        q_s = query_start - 1
                        q_e = query_end + 1
                        q_s = max(0, q_s)
                        q_e = min(text_encoded.shape[0], q_e)
                        # breakpoint()
                    else:
                        q_s = query_start
                        q_e = query_end

                    text_enc_pooled = text_encoded[q_s:q_e]

                    # select end_lat - start_lat equally spaced frames from text_enc_pooled
                    remainder = text_enc_pooled.shape[0] % (end_lat - start_lat)
                    if remainder > 0 and text_enc_pooled.shape[0] > (end_lat - start_lat):
                        text_enc_pooled = text_enc_pooled[:-remainder]

                    if text_enc_pooled.shape[0] // (end_lat - start_lat) == 0:
                        # if text_enc_pooled.shape[0] == 0:
                        #     breakpoint()

                        text_enc_pooled = text_enc_pooled[:1].expand(end_lat - start_lat, -1)
                    else:
                        text_enc_pooled = text_enc_pooled[::text_enc_pooled.shape[0] // (end_lat - start_lat)]


                    zero_text[start_lat:end_lat] = text_enc_pooled
                    zero_text[latent_len + 1 + start_lat: latent_len + 1 + end_lat] = text_enc_pooled
                    zero_text[2 * latent_len + 2 + start_lat: 2 * latent_len + 2 + end_lat] = text_enc_pooled
                    zero_text[3 * latent_len + 3 + start_lat: 3 * latent_len + 3 + end_lat] = text_enc_pooled
                    zero_raw_text[start_lat*self.motion_framechunksize:end_lat*self.motion_framechunksize] = text_enc_pooled.repeat(self.motion_framechunksize, 1)
                    
            
            all_type2words.append(batch_type2words)
            all_masked_motions.append(zero_motion)
            all_words.append(zero_text)
            all_raw_words.append(zero_raw_text)
            raw_masked_motions.append(zero_raw_motion)
            raw_masked_motions_aa.append(zero_raw_motion_aa)
            raw_masked_trans.append(zero_raw_trans)
            raw_masked_facial.append(zero_raw_facial)

            all_retr_startends.append(retr_startend)
            all_query_startends.append(query_startend)
            all_retr_latents.append(retrlatents_uncropped)

        N = len(all_masked_motions)
        all_motions = torch.stack(all_masked_motions, dim=0).to(device)
        
        

        all_raw_motions = torch.stack(raw_masked_motions_aa, dim=0).to(device)
        all_raw_trans = torch.stack(raw_masked_trans, dim=0).to(device)
        all_raw_facial = torch.stack(raw_masked_facial, dim=0).to(device)

        # getting the sample names for the retrieved motions for future reference
        all_sample_names = []
        # all_type2words_ = []
        for b_ix in range(B):
            type2words = all_type2words[b_ix]
            all_sample_names.append({})
            for query_point_idx, smp_idxs in all_indexes[b_ix].items():
                if query_point_idx not in type2words:
                    continue
                q_word, q_type, r_word, r_type = type2words[query_point_idx]

                assert len(smp_idxs) == self.num_retrieval == 1
                all_sample_names[-1][q_word] = self.dataset[smp_idxs[0]]["sample_name"]

        # text feature processing:
        # breakpoint() # check all words shape
        all_text_features = torch.stack(all_words, dim=0).to(device)
        

        T = all_text_features.shape[1]
        # breakpoint()  # check all_text_features shape and self.text_pos_embedding shape
        # T should be equal to self.text_pos_embedding.shape[0]
        

        # motion feature processing:
        T = all_motions.shape[1] 
        src_mask = (all_motions != 0).any(dim=-1).to(torch.int).to(device)
        raw_latent_mask = src_mask.clone() # TODO: check how to change this

        all_motions_reshaped = all_motions
        
        raw_motion_latents = all_motions_reshaped.clone()

        upper_indices = list(range(0, (T-3)//4))
        hands_indices = list(range((T-3)//4 + 1, 2*(T-3)//4 + 1))
        face_indices = list(range(2*(T-3)//4 + 2, 3*(T-3)//4 + 2))
        lowertrans_indices = list(range(3*(T-3)//4 + 3, T))

        
        src_mask[:, face_indices + lowertrans_indices] = 0
        raw_motion_latents[:, face_indices + lowertrans_indices, :] = 0
        

        # breakpoint()  # check the raw motion and raw text features
        raw_motion = all_raw_motions.view(B, self.num_retrieval, self.max_seq_len, -1).contiguous()
        raw_trans = all_raw_trans.view(B, self.num_retrieval, self.max_seq_len, -1).contiguous()
        raw_facial = all_raw_facial.view(B, self.num_retrieval, self.max_seq_len, 100).contiguous()
        raw_motion_latents = raw_motion_latents.view(B, self.num_retrieval, T, -1).contiguous()
        # breakpoint()  # check the raw motion and raw text features

        re_dict = dict(
            re_text=None,
            re_motion=None,
            
            re_mask=src_mask,
            raw_motion_latents=raw_motion_latents,
            raw_motion=raw_motion,
            raw_trans=raw_trans,
            raw_facial=raw_facial,
            raw_sample_names=all_sample_names,
            raw_type2words=all_type2words,
            raw_latent_mask=raw_latent_mask, # without upper selection

            retr_startends=all_retr_startends,
            query_startends=all_query_startends,
            retr_uncropped_latents=all_retr_latents,
        )
        return re_dict


@SUBMODULES.register_module()
class ReGestureTransformer(DiffusionTransformer):
    def __init__(self, 
                 retrieval_cfg=None, 
                 scale_func_cfg=None, 
                 per_joint_scale=None, 
                 retrieval_train=False, 
                 use_retrieval_for_test=False, 
                 **kwargs
                 ):
        
        dataset = kwargs.pop("database")
        super().__init__(**kwargs)
        assert not retrieval_train
        if retrieval_cfg is not None and use_retrieval_for_test:
            current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.database = RetrievalDatabase(
                **retrieval_cfg,
                dataset=dataset,
                device=current_device,
            )
        else:
            self.database = None
        self.scale_func_cfg = scale_func_cfg

        self.reward_adapter = None


        self.per_joint_scale = per_joint_scale
        if self.per_joint_scale is not None:
            T = 43
            upper_indices = list(range(0, (T-3)//4))
            hands_indices = list(range((T-3)//4 + 1, 2*(T-3)//4 + 1))
            face_indices = list(range(2*(T-3)//4 + 2, 3*(T-3)//4 + 2))
            lowertransl_indices = list(range(3*(T-3)//4 + 3, T))

            self.joint_scale_mask = torch.ones(T)
            self.joint_scale_mask[upper_indices] = self.per_joint_scale["upper"]
            self.joint_scale_mask[hands_indices] = self.per_joint_scale["hands"]
            self.joint_scale_mask[face_indices] = self.per_joint_scale["face"]
            self.joint_scale_mask[lowertransl_indices] = self.per_joint_scale["lowertransl"]

    def load_reward_adapter(self, adapter_path, vae_latent_dim=None):
        """
        adapter_path: 权重文件路径 (.pth)
        vae_latent_dim: 如果不传，尝试使用 self.latent_dim 或默认值
        """
        
            
        print(f"[RAG-Gesture] Loading Reward Adapter from {adapter_path} (Input Dim: {vae_latent_dim})...")
        
        # 实例化 
        self.reward_adapter = StepAwareAdapter(input_dim=512, output_dim=256)
        
        # 加载权重
        ckpt = torch.load(adapter_path, map_location='cpu')
        self.reward_adapter.load_state_dict(ckpt)
        
        # 移动到当前设备的 GPU 并冻结
        device = next(self.parameters()).device
        self.reward_adapter.to(device).eval()
        for p in self.reward_adapter.parameters():
            p.requires_grad = False
    
    def calc_segment_guidance(self, x_latent, t, target_emb, start_frame, end_frame):
        """
        计算局部时间窗口的梯度引导。
        x_latent: [B, SeqLen, Dim] 当前全序列动作
        t: [B] 时间步
        target_emb: [B, 512] 目标语义向量 (Text 或 Ref Motion)
        start_frame, end_frame: 切片索引
        """
        if self.reward_adapter is None:
            return torch.zeros_like(x_latent)

        # 1. 开启梯度记录 (因为 x_latent 通常是 detach 的，这里要重新挂载梯度)
        with torch.enable_grad():
            x_in = x_latent.detach().requires_grad_(True)
            
            # 2. [Crop] 裁剪：只取出需要引导的片段
            # 注意：请确保 start/end 不越界
            seq_len = x_in.shape[1]
            s = max(0, start_frame)
            e = min(seq_len, end_frame)
            
            if s >= e: # 如果片段无效，返回0梯度
                return torch.zeros_like(x_latent)

            x_segment = x_in[:, s:e, :] # [B, SegLen, Dim]
            
            # 3. [Predict] Adapter 预测语义
            # Adapter 需要处理变长序列，或者您确保输入的 shape 符合 Adapter 预期
            pred_emb = self.reward_adapter(x_segment, t)
            
            # 4. [Loss] 计算与目标的相似度 (Goal: Maximize Cosine Similarity)
            # Loss = -Cosine (因为我们要梯度下降)
            loss = -torch.cosine_similarity(pred_emb, target_emb, dim=-1).sum()
            
            # 5. [Gradient] 对 x_in 求导
            # Autograd 会自动处理切片的反向传播，只计算 s:e 部分的梯度
            grad = torch.autograd.grad(loss, x_in)[0]
            
            # 6. [Paste/Mask] 双重保险：强制将非引导区域的梯度置零
            # 虽然 autograd 理论上已经处理了，但手动 mask 更安全，防止数值噪音
            mask = torch.zeros_like(grad)
            mask[:, s:e, :] = 1.0
            final_grad = grad * mask
            
        return final_grad
    def scale_func_retr(self, timestep):
        coarse_scale = self.scale_func_cfg["coarse_scale"]
        w = (1 - (1000 - timestep) / 1000) * coarse_scale + 1
        if timestep > 100:
            if random.randint(0, 1) == 0:
                output = {
                    "both_coef": w,
                    "text_coef": 0,
                    "retr_coef": 1 - w,
                    "none_coef": 0,
                }
            else:
                output = {
                    "both_coef": 0,
                    "text_coef": w,
                    "retr_coef": 0,
                    "none_coef": 1 - w,
                }
        else:
            both_coef = self.scale_func_cfg["both_coef"]
            text_coef = self.scale_func_cfg["text_coef"]
            retr_coef = self.scale_func_cfg["retr_coef"]
            none_coef = 1 - both_coef - text_coef - retr_coef
            output = {
                "both_coef": both_coef,
                "text_coef": text_coef,
                "retr_coef": retr_coef,
                "none_coef": none_coef,
            }
        return output


    def get_precompute_condition(
        self,
        text=None,
        raw_text=None,
        text_features=None,
        audio=None,
        raw_audio=None,
        discourse=None,
        prominence=None,
        speaker_ids=None,
        gesture_labels=None,
        text_times=None,
        motion_length=None,
        xf_out=None,
        re_dict=None,
        device=None,
        sample_idx=None,
        sample_name=None,
        retrieval_method="gesture_type",
        **kwargs,
    ):
        if xf_out is None:
            xf_out_text = self.encode_text(text, device)
            xf_out_audio = self.encode_audio(audio, device)
            xf_spk = self.encode_spks(speaker_ids, device)
            
            xf_out = {
                "xf_text": xf_out_text,
                "xf_audio": xf_out_audio,
                "xf_spk": xf_spk,
            }
        output = {"xf_out": xf_out}
        
        if re_dict is None and self.database is not None:
            retr_conditions = dict(
                text=raw_text, 
                audio=raw_audio, 
                text_enc=text,
                text_features=text_features,
                audio_enc=audio,
                discourse=discourse,
                prominence=prominence,
                speaker_ids=speaker_ids,
                gesture_labels=gesture_labels,
                text_times=text_times,
            )
            re_dict = self.database(
                retr_conditions,
                motion_length,
                device,
                idx=sample_name,  # sample_idx
                llm_full_context = kwargs.get("llm_full_context", ""),
                llm_focus_window=kwargs.get("llm_focus_window", None),
                retrieval_method=retrieval_method,
                gesture_rep_encoder=self.gesture_rep_encoder,
            )
        output["re_dict"] = re_dict
        
        return output

    def post_process(self, motion):
        return motion

    def forward_train(
        self, h=None, src_mask=None, emb=None, xf_out=None, query_mask=None, re_dict=None, **kwargs
    ):
        start = time.time()
        B, T = h.shape[0], h.shape[1]
        
        cond_type = torch.randint(0, 100, size=(B, 1, 1)).to(h.device)
        for module in self.temporal_decoder_blocks:
            h = module(
                x=h,
                xf=xf_out,
                emb=emb,
                src_mask=src_mask,
                query_mask=query_mask,
                cond_type=cond_type,
                re_dict=re_dict,
            )

        
        output = self.out(h).view(B, T, -1).contiguous()
        
        return output, re_dict
    
    def forward_test(
        self,
        h=None,
        src_mask=None,
        emb=None,
        xf_out=None,
        query_mask=None,
        timesteps=None,
        do_clf_guidance=False,
        **kwargs,
    ):
        
        B, T = h.shape[0], h.shape[1]
        text_cond_type = torch.zeros(B, 1, 1).to(h.device) + 1
        if do_clf_guidance or self.scale_func_cfg is not None:
            none_cond_type = torch.zeros(B, 1, 1).to(h.device)

            cond_types = (text_cond_type, none_cond_type)
            all_cond_type = torch.cat(cond_types, dim=0)

            h = h.repeat(len(cond_types), 1, 1)
            xf_out = {
                k: v.repeat(len(cond_types), 1, 1) for k, v in xf_out.items()
            }
            emb = emb.repeat(len(cond_types), 1)
            src_mask = src_mask.repeat(len(cond_types), 1, 1)

            if query_mask is not None:
                for k, v in query_mask.items():
                    query_mask[k] = v.repeat(len(cond_types), 1)

        else:
            all_cond_type = text_cond_type
        
        
        for module in self.temporal_decoder_blocks:
            h = module(
                x=h,
                xf=xf_out,
                emb=emb,
                src_mask=src_mask,
                query_mask=query_mask,
                cond_type=all_cond_type,
            )
        
        out = self.out(h)
        if do_clf_guidance or self.scale_func_cfg is not None:
            out = out.view(2 * B, T, -1).contiguous()
            
            if self.scale_func_cfg is not None:

                coef_cfg = self.scale_func_retr(int(timesteps[0]))
                both_coef = coef_cfg["both_coef"]
                text_coef = coef_cfg["text_coef"]
                retr_coef = coef_cfg["retr_coef"]
                none_coef = coef_cfg["none_coef"]

                out_text = out[:B].contiguous()
                out_none = out[B : 2 * B].contiguous()
                
                
                joint_scale_tensor = self.joint_scale_mask.unsqueeze(0).unsqueeze(-1).expand(B, -1, out_text.shape[-1])
                joint_scale_tensor = joint_scale_tensor.to(out_text.device)

                output = (
                    out_text * both_coef * joint_scale_tensor
                    + out_text * text_coef * joint_scale_tensor
                    + out_none * retr_coef * (1/joint_scale_tensor)
                    + out_none * none_coef * (1/joint_scale_tensor)
                )
                out = output

        return out
