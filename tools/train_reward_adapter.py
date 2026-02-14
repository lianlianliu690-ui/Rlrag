import argparse
import os
import torch

import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from mmcv import Config
import yaml
from functools import partial

import os
import sys

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from mogen.models.transformers.diffusion_transformer import GestureRepEncoder
from mogen.models.utils.gaussian_diffusion import (
    GaussianDiffusion,
    get_named_beta_schedule,
    ModelMeanType,
    ModelVarType,
    LossType
)
# === 导入你的项目模块 ===
from mogen.models.transformers.reward_adapter import StepAwareAdapter
from mogen.datasets import build_dataset, build_dataloader 
from mogen.models.builder import build_architecture # 用于构建 VAE
from build_motion_layer import TMRMotionWrapper
from mogen.models.transformers.gesture_vae import TransformerVAE
from mogen.datasets.builder import beatx_collate_fn 


# ==========================================
# 训练主逻辑
# ==========================================
def train_adapter(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载配置
    cfg = Config.fromfile(args.config)
    
    # 2. 构建 Dataset
    print("Building Dataset...")
    dataset = build_dataset(cfg.data.train)
    dataloader = build_dataloader(
        dataset,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,
        num_gpus=1,  # 单GPU训练
        dist=False,  # 非分布式训练
        shuffle=True,
        round_up=True,
        seed=cfg.get('seed', None)
    )

    # 3. 加载 VAE (Gesture Rep Encoder)
    print("Loading VAE (Gesture Rep Encoder)...")
    try:
        if 'model' in cfg.model and 'vae_cfg' in cfg.model.model:
            # 这里的路径是 cfg -> model -> model -> vae_cfg
            vae_config = cfg.model.model.vae_cfg
            # 同时获取拼接轴配置
            body_part_cat_axis = cfg.model.model.get('body_part_cat_axis', 'time')

            
        print(f"Found VAE Config. Body part concat axis: {body_part_cat_axis}")
        
    except Exception as e:
        print(f"Config parsing error: {e}")
        print("Please check if your config file structure matches: model = dict(model=dict(vae_cfg=...))")
        raise e

    # [关键] 实例化 VAE
    # GestureRepEncoder 内部会读取 vae_config 中的 yaml 路径并加载各个部位的权重
    vae_model = GestureRepEncoder(vae_config, body_part_cat_axis)
    
    # 注意：这里不再需要 load_state_dict(args.vae_ckpt)，因为权重已经在各部位初始化时加载了
    
    vae_model.to(device).eval()
    for p in vae_model.parameters(): p.requires_grad = False
    
    # 获取 Latent Dim
    # 通常从 config 的 latent_dim 读取，或者从模型属性读取
    if 'latent_dim' in vae_config:
        vae_latent_dim = vae_config['latent_dim']       #512
    else:
        vae_latent_dim = getattr(vae_model, 'vae_latent_dim', 32)
        
    print(f"VAE Latent Dim (Student Input): {vae_latent_dim}")

    
    # 4. 加载 Teacher (TMR)
    print("Loading Teacher (TMR)...")
    # 请替换为你实际的 TMR 路径
    tmr_path = "/Dataset4D/public/mas-liu.lianlian/pretrained_models/beatx_1-30_amass_h3d_tmr/"
    teacher_model = TMRMotionWrapper(tmr_path, device)

    # 5. 初始化 Adapter
    print("Initializing Reward Adapter...")
    student_model = StepAwareAdapter(
        input_dim=vae_latent_dim, 
        output_dim=256            
    ).to(device)
    student_model.train()

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4)

    # =========================================================
    # 6. 初始化 Gaussian Diffusion (严格对齐 Config)
    # =========================================================
    print("Initializing Gaussian Diffusion Scheduler...")
    
    # 从您的 config.model.diffusion_train 中读取
    # 注意：config 文件里是 model.diffusion_train 字典
    diff_cfg = cfg.model.diffusion_train
    
    beta_scheduler = diff_cfg.get('beta_scheduler', 'linear') # 您的是 'scaled_linear'
    diffusion_steps = diff_cfg.get('diffusion_steps', 1000)     # 1000
    
    print(f"Diffusion Config: Steps={diffusion_steps}, Schedule={beta_scheduler}")
    
    # 1. 获取 betas (复刻 build_diffusion 的逻辑)
    betas = get_named_beta_schedule(beta_scheduler, diffusion_steps)
    
    # 2. 解析 Mean Type (您的是 start_x)
    # 虽然 q_sample 不需要 mean_type，但为了保持对象完整性，建议填对
    model_mean_type_str = diff_cfg.get('model_mean_type', 'epsilon')
    model_mean_type = {
        "start_x": ModelMeanType.START_X,
        "previous_x": ModelMeanType.PREVIOUS_X,
        "epsilon": ModelMeanType.EPSILON,
        "v_pred": ModelMeanType.V_PRED,
    }[model_mean_type_str]

    # 3. 解析 Var Type (您的是 fixed_large)
    model_var_type_str = diff_cfg.get('model_var_type', 'fixed_small')
    model_var_type = {
        "learned": ModelVarType.LEARNED,
        "fixed_small": ModelVarType.FIXED_SMALL,
        "fixed_large": ModelVarType.FIXED_LARGE,
        "learned_range": ModelVarType.LEARNED_RANGE,
    }[model_var_type_str]

    # 4. 实例化
    diffusion = GaussianDiffusion(
        betas=betas, # 关键！传入计算好的 betas
        model_mean_type=model_mean_type,
        model_var_type=model_var_type,
        loss_type=LossType.MSE, # 您的 config 没写，默认 MSE
        rescale_timesteps=False, # 您的代码似乎没开这个
    )
    
    # 7. 训练循环
    print("Start Training...")
    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            # === A. 准备数据 ===
            m_upper = batch['motion_upper'].to(device)
            m_lower = batch['motion_lower'].to(device)
            m_hands = batch['motion_hands'].to(device)
            m_face  = batch['motion_face'].to(device)
            m_trans = batch['trans'].to(device)
            m_facial= batch['facial'].to(device)
            m_cont  = batch['contact'].to(device)
            m_mask  = batch['motion_mask'].to(device)
            
            # Teacher Input
            # Note: motion_h3d contains variable-length sequences, processed individually below
            m_h3d = batch['motion_h3d'].to(device)


            B = m_upper.shape[0]

            # === B. Teacher Path (生成目标) ===
            if 'motion_h3d' in batch:
                target_feat = teacher_model(m_h3d) # [B, TMR_OUTPUT_DIM]
            else:
                with torch.no_grad():
                    target_feat = teacher_model(m_h3d)

            # === C. VAE Path (生成学生输入) ===
            with torch.no_grad():
                latents_tuple = vae_model.encode(
                    m_upper, m_lower, m_face, m_hands, m_trans, m_facial, m_cont, m_mask
                )
                full_latents = latents_tuple[0] 
                
            # === D. Student Path (真实加噪预测) ===
            # 1. 随机采样时间步 t
            t = torch.randint(0, diffusion_steps, (B,), device=device).long()
            
            # 2. 生成噪声
            noise = torch.randn_like(full_latents)
            
            # 3. [关键] 使用 GaussianDiffusion.q_sample 进行加噪
            # x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise
            noisy_latents = diffusion.q_sample(x_start=full_latents, t=t, noise=noise)
            
            # Adapter 预测
            pred_feat = student_model(noisy_latents, t)

            # === E. Loss ===
            loss_mse = F.mse_loss(pred_feat, target_feat)
            loss_cos = 1 - F.cosine_similarity(pred_feat, target_feat).mean()
            
            loss = loss_mse + 0.1 * loss_cos
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({'loss': loss.item()})

        if (epoch + 1) % 5 == 0:
            os.makedirs("work_dirs/reward_adapter", exist_ok=True)
            torch.save(student_model.state_dict(), f"work_dirs/reward_adapter/epoch_{epoch+1}.pth")

if __name__ == '__main__':
    # import debugpy
    # try:
    #     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    #     debugpy.listen(("localhost", 9502))
    #     print("Waiting for debugger attach")
    #     debugpy.wait_for_client()
    # except Exception as e:
    #   pass
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    # parser.add_argument('--vae_ckpt', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    
    train_adapter(args)