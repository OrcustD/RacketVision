import torch
import torch.nn as nn
from model.loss import weighted_mse_loss
from mmengine.registry import MODELS
from mmengine.model import BaseModel
from mamba_ssm import Mamba

@MODELS.register_module(name='TPMambaformer')
class TrajectoryPredictorMambaformer(BaseModel):
    """
    一个基于Mamba和跨注意力融合的轨迹预测模型。
    """
    def __init__(self,
                 pred_len=20,
                 ball_input_dim=2,
                 racket_input_dim=10,
                 # Model dimensions
                 d_model=512,
                 # Cross-Attention parameters
                 nhead=8,
                 # Mamba parameters
                 num_mamba_layers=4,
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 # Other parameters
                 dropout=0.1,
                 min_loss_weight=0.5):
        
        super().__init__()

        if Mamba is None:
            raise ImportError("Mamba model requires the 'mamba-ssm' package. Please install it.")

        self.pred_len = pred_len
        self.loss_weights = torch.linspace(1.0, min_loss_weight, steps=pred_len)

        # --- 1. 定义独立的嵌入层 ---
        self.ball_embed = nn.Linear(ball_input_dim, d_model)
        self.racket_embed = nn.Linear(racket_input_dim, d_model)
        
        # --- 2. 定义跨注意力融合模块 ---
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=nhead, 
            dropout=dropout, 
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # --- 3. 定义Mamba核心时序建模模块 ---
        mamba_layers = []
        for _ in range(num_mamba_layers):
            mamba_layers.append(
                Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand
                )
            )
        self.mamba_layers = nn.Sequential(*mamba_layers)
        
        # --- 4. 定义输出解码器 ---
        self.decoder = nn.Linear(d_model, self.pred_len * 2)

    def _forward(self, ball_hist, racket_hist):
        """
        核心前向传播逻辑，实现了"Attention pre-fusion + Mamba processing"。
        """
        # 1. 特征嵌入
        ball_embedded = self.ball_embed(ball_hist)
        racket_embedded = self.racket_embed(racket_hist)
        
        # 注意：Mamba不需要位置编码
        ball_seq = ball_embedded
        racket_seq = racket_embedded

        # 2. 跨注意力融合
        attended_ball_features, _ = self.cross_attention(
            query=ball_seq, key=racket_seq, value=racket_seq)
        
        # 3. 残差连接和层归一化
        fused_seq = self.norm1(ball_seq + self.dropout1(attended_ball_features))
        
        # 4. 通过Mamba层进行时序建模
        mamba_output = self.mamba_layers(fused_seq)
        
        # 5. 解码
        last_time_step_out = mamba_output[:, -1, :]
        prediction_flat = self.decoder(last_time_step_out)
        prediction = prediction_flat.view(-1, self.pred_len, 2)
        
        return prediction

    def forward(self, history=None, future=None, history_rkt=None, future_rkt=None, metadata=None, mode='predict'):
        """
        完整的 MMEngine BaseModel forward 函数。
        """
        if history_rkt is None:
            raise ValueError("Mambaformer model requires racket history ('history_rkt').")
            
        x = self._forward(history, history_rkt)
        labels = future
        
        if mode == 'loss':
            return {'loss': weighted_mse_loss(x, labels, self.loss_weights)}
        if mode == 'predict':
            return x, history, future, history_rkt, metadata
        else:
            raise ValueError(f"Unknown mode: {mode}")