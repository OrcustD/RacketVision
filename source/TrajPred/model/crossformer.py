import torch
import torch.nn as nn
import math
from model.loss import weighted_mse_loss
from mmengine.registry import MODELS
from mmengine.model import BaseModel

# PositionalEncoding 类与之前完全相同，这里省略以保持简洁
# 您可以从上一个回答中直接复制过来使用
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 假设输入为 (N, S, E)
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

@MODELS.register_module(name='TPCrossformer')
class TrajectoryPredictorCrossformer(BaseModel):
    """
    一个基于跨注意力融合的Transformer轨迹预测模型。
    这个模型专门用于 'ball_racket' 模式。
    """
    def __init__(self,
                 pred_len=20,
                 ball_input_dim=2,
                 racket_input_dim=10,
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=4, # 建议使用更深的模型
                 dim_feedforward=2048,
                 dropout=0.1,
                 min_loss_weight=0.5,
                 alpha=0.1):
        super().__init__()
        
        self.pred_len = pred_len
        self.loss_weights = torch.linspace(1.0, min_loss_weight, steps=pred_len)
        self.alpha = alpha

        # --- 1. 定义独立的嵌入层 ---
        # 将球和球拍都嵌入到相同的 d_model 维度
        self.ball_embed = nn.Linear(ball_input_dim, d_model)
        self.racket_embed = nn.Linear(racket_input_dim, d_model)
        
        # --- 2. 定义核心模块 ---
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 跨注意力层
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=nhead, 
            dropout=dropout, 
            batch_first=True
        )
        
        # 用于处理融合后特征的自注意力编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.self_attention_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        
        # --- 3. 定义层归一化和Dropout ---
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # --- 4. 定义输出解码器 ---
        self.decoder = nn.Linear(d_model, self.pred_len * 2)

    def _forward(self, ball_hist, racket_hist):
        """
        核心前向传播逻辑，实现了跨注意力融合。
        """
        # 1. 特征嵌入: (N, S, Cin) -> (N, S, d_model)
        ball_embedded = self.ball_embed(ball_hist)
        racket_embedded = self.racket_embed(racket_hist)
        
        # 2. 添加位置编码
        ball_seq = self.pos_encoder(ball_embedded)
        racket_seq = self.pos_encoder(racket_embedded)

        # 3. 跨注意力融合
        # ball_seq 作为 Query, racket_seq 作为 Key 和 Value
        # 球"查询"球拍信息来增强自己
        attended_ball_features, _ = self.cross_attention(
            query=ball_seq, 
            key=racket_seq, 
            value=racket_seq
        )
        
        # 4. 第一个残差连接和层归一化 (Add & Norm)
        # fused_seq = ball_seq + self.dropout1(attended_ball_features)
        # fused_seq = self.norm1(fused_seq)
        fused_seq = self.norm1(ball_seq + self.alpha * self.dropout1(attended_ball_features))
        
        # 5. 通过自注意力编码器进行时序建模
        encoder_output = self.self_attention_encoder(fused_seq)
        
        # 6. 解码
        last_time_step_out = encoder_output[:, -1, :]
        prediction_flat = self.decoder(last_time_step_out)
        prediction = prediction_flat.view(-1, self.pred_len, 2)
        
        return prediction

    def forward(self, history=None, future=None, history_rkt=None, future_rkt=None, metadata=None, mode='predict'):
        """
        完整的 MMEngine BaseModel forward 函数。
        """
        if history_rkt is None:
            raise ValueError("Crossformer model requires racket history ('history_rkt').")
            
        # 调用核心逻辑
        x = self._forward(history, history_rkt)
        labels = future
        
        if mode == 'loss':
            return {'loss': weighted_mse_loss(x, labels, self.loss_weights)}
        if mode == 'predict':
            return x, history, future, history_rkt, metadata
        else:
            raise ValueError(f"Unknown mode: {mode}")