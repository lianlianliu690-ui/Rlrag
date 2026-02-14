import torch
import torch.nn as nn
import math

class StepAwareAdapter(nn.Module):
    def __init__(self, input_dim, output_dim=512, hidden_dim=512, num_layers=4, nhead=4):
        """
        input_dim: VAE 的 Latent Dim (不是原始动作维度 263，而是 VAE 压缩后的维度，例如 32)
        output_dim: TMR/SynTalker 的输出维度 (通常是 512)
        """
        super().__init__()
        
        # 1. 动作投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 2. 时间步嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=1024, 
            batch_first=True,
            activation='gelu',
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. 输出投影
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        
        # 5. Attention Pooling (自适应聚合)
        self.aggregator = nn.Parameter(torch.randn(1, 1, hidden_dim))



    def forward(self, x, t, mask=None):
        # x shape: [B, 4*T_latent, Latent_Dim] (因为包含 Upper, Hands, Face, Lower)
        B, L, _ = x.shape
        
        x_emb = self.input_proj(x)
        t_emb = self.time_mlp(t).unsqueeze(1)
        h = x_emb + t_emb
        
        output = self.transformer(h) # [B, L, H]
        
        # Attention Pooling
        query = self.aggregator.expand(B, -1, -1) # [B, 1, H]
        scores = torch.bmm(query, output.transpose(1, 2)) # [B, 1, L]
        
        if mask is not None:
             scores = scores.masked_fill(mask.unsqueeze(1), -1e9)
             
        attn_weights = torch.softmax(scores, dim=-1)
        feature = torch.bmm(attn_weights, output).squeeze(1) # [B, H]
            
        return self.out_proj(feature)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb