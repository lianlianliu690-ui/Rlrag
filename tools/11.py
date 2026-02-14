# ... existing code ...
import torch
import numpy as np
from mmcv import Config
import os
import sys
import warnings

# 添加必要的路径到Python路径
sys.path.insert(0, '/home/mas-liu.lianlian/RLrag')
from mogen.datasets.builder import build_dataset
import os

def scan_samples_for_motion_upper_length(dataset_config_path, start_idx=0, end_idx=None):
    """
    遍历样本，找出motion_upper长度不是150的样本
    """
    # 加载数据集配置
    cfg = Config.fromfile(dataset_config_path)
    
    # 构建数据集
    dataset = build_dataset(cfg.data.train)
    
    total_samples = len(dataset)
    print(f"数据集总样本数: {total_samples}")
    
    if end_idx is None:
        end_idx = total_samples
    
    # 确保索引范围有效
    end_idx = min(end_idx, total_samples)
    
    print(f"扫描范围: {start_idx} 到 {end_idx}")
    
    invalid_samples = []
    
    for i in range(start_idx, end_idx):
        try:
            sample = dataset[i]
            
            
            if 'motion_h3d' in sample:
                motion_upper = sample['motion_h3d']
                
                if isinstance(motion_upper, torch.Tensor):
                    length = motion_upper.shape[0]
                elif isinstance(motion_upper, np.ndarray):
                    length = motion_upper.shape[0]
                else:
                    print(f"样本 {i} 的 motion_upper 不是张量或数组，类型: {type(motion_upper)}")
                    continue
                
                if length != 150:
                    invalid_samples.append((i, length))
                    print(f"发现异常样本: ID={i}, motion_upper长度={length}")
                    
                    # 同时显示其他相关信息
                    if 'motion' in sample:
                        motion_len = sample['motion'].shape[0] if isinstance(sample['motion'], torch.Tensor) else sample['motion'].shape[0]
                        print(f"  - motion长度: {motion_len}")
                    if 'motion_h3d' in sample:
                        h3d_len = sample['motion_h3d'].shape[0] if isinstance(sample['motion_h3d'], torch.Tensor) else sample['motion_h3d'].shape[0]
                        print(f"  - motion_h3d长度: {h3d_len}")
                    if 'sample_name' in sample:
                        print(f"  - sample_name: {sample['sample_name']}")
                    print("-" * 50)
            
        except Exception as e:
            print(f"读取样本 {i} 时发生错误: {str(e)}")
            continue
        
        # 每1000个样本打印一次进度
        if (i + 1) % 1000 == 0:
            print(f"已扫描 {i + 1}/{end_idx} 个样本")
    
    print(f"\n扫描完成！共发现 {len(invalid_samples)} 个motion_upper长度不是150的样本:")
    for sample_id, length in invalid_samples:
        print(f"  样本ID: {sample_id}, 长度: {length}")
    
    return invalid_samples

def main():
    # 设置数据集配置文件路径
    dataset_config_path = "/home/mas-liu.lianlian/RLrag/configs/raggesture_beatx/basegesture_len150_beat.py"
    
    # 检查配置文件是否存在
    if not os.path.exists(dataset_config_path):
        print(f"配置文件不存在: {dataset_config_path}")
        return
    
    # 扫描所有样本
    invalid_samples = scan_samples_for_motion_upper_length(dataset_config_path)
    
    if invalid_samples:
        print(f"\n总结: 发现 {len(invalid_samples)} 个异常样本")
    else:
        print("\n所有样本的 motion_upper 长度都是150")

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
# ... existing code ...