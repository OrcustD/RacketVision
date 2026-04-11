import torch
import torch.nn as nn
import math
from model.loss import weighted_mse_loss
from mmengine.registry import MODELS
from mmengine.model import BaseModel

class PositionalEncoding(nn.Module):
    """
    为Transformer注入序列的位置信息。
    """
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
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # Pytorch TransformerEncoder默认输入为(S, N, E)，如果batch_first=True则为(N, S, E)
        # 我们的 PositionalEncoding 假设输入为 (N, S, E)
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

@MODELS.register_module(name='TPTransformer')
class TrajectoryPredictorTransformer(BaseModel):
    """
    一个基于Transformer的轨迹预测模型（简单特征拼接版）。
    """
    def __init__(self,
                 mode='ball_only',
                 pred_len=20,
                 # Ball embedding dimensions
                 ball_input_dim=2,
                 ball_embed_dim=32,
                 # Racket embedding dimensions
                 racket_input_dim=10,
                 racket_embed_dim=96,
                 # Transformer parameters
                 d_model=128,
                 nhead=8,
                 num_encoder_layers=3,
                 dim_feedforward=512,
                 dropout=0.1,
                 min_loss_weight=0.5):
        """
        初始化模型.
        
        Args:
            mode (str): 'ball_only' 或 'ball_racket'.
            pred_len (int): 需要预测的未来轨迹长度.
            ball_embed_dim (int): 球坐标嵌入后的维度.
            racket_embed_dim (int): 球拍姿态嵌入后的维度.
            d_model (int): Transformer的核心维度，必须等于 ball_embed_dim (+ racket_embed_dim)。
            nhead (int): Transformer中的多头注意力头数。
            num_encoder_layers (int): Transformer编码器的层数。
            dim_feedforward (int): Transformer中前馈网络的维度。
            dropout (float): Dropout概率。
            min_loss_weight (float): 最小的损失权重.
        """
        super().__init__()
        
        if mode not in ['ball_only', 'ball_racket']:
            raise ValueError("Mode must be 'ball_only' or 'ball_racket'")
        
        self.mode = mode
        self.pred_len = pred_len
        self.loss_weights = torch.linspace(1.0, min_loss_weight, steps=pred_len)

        # --- 1. 定义嵌入层 ---
        if self.mode == 'ball_only':
            # 确保输入嵌入的维度等于d_model
            assert ball_embed_dim == d_model, "In 'ball_only' mode, ball_embed_dim must be equal to d_model."
            self.ball_embed = nn.Linear(ball_input_dim, d_model)
        else: # ball_racket mode
            # 确保拼接后的维度等于d_model
            assert ball_embed_dim + racket_embed_dim == d_model, "In 'ball_racket' mode, ball_embed_dim + racket_embed_dim must equal d_model."
            self.ball_embed = nn.Linear(ball_input_dim, ball_embed_dim)
            self.racket_embed = nn.Linear(racket_input_dim, racket_embed_dim)
        
        # --- 2. 定义Transformer核心 ---
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True  # 重要！确保输入shape为 (N, S, E)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        
        # --- 3. 定义输出解码器 ---
        self.decoder = nn.Linear(d_model, self.pred_len * 2)

    def _forward(self, ball_hist, racket_hist=None):
        """
        核心前向传播逻辑.
        """
        # --- 1. 特征嵌入和融合 ---
        if self.mode == 'ball_only':
            # (N, S, 2) -> (N, S, d_model)
            embedded_input = self.ball_embed(ball_hist)
        else: # ball_racket mode
            if racket_hist is None:
                raise ValueError("Racket history must be provided in 'ball_racket' mode.")
            # (N, S, 2) -> (N, S, ball_embed_dim)
            embedded_ball = self.ball_embed(ball_hist)
            # (N, S, 10) -> (N, S, racket_embed_dim)
            embedded_racket = self.racket_embed(racket_hist)
            # 拼接: (N, S, d_model)
            embedded_input = torch.cat([embedded_ball, embedded_racket], dim=-1)

        # --- 2. 通过Transformer处理序列 ---
        # 注入位置信息
        pos_encoded_input = self.pos_encoder(embedded_input)
        # Transformer编码
        transformer_out = self.transformer_encoder(pos_encoded_input)
        
        # --- 3. 解码预测未来轨迹 ---
        # 使用最后一个时间步的输出来进行预测
        last_time_step_out = transformer_out[:, -1, :]
        
        prediction_flat = self.decoder(last_time_step_out)
        prediction = prediction_flat.view(-1, self.pred_len, 2)
        
        return prediction

    def forward(self, history=None, future=None, history_rkt=None, future_rkt=None, metadata=None, mode='predict'):
        """
        完整的 MMEngine BaseModel forward 函数。
        """
        # 调用核心逻辑
        x = self._forward(history, history_rkt)
        labels = future
        
        if mode == 'loss':
            return {'loss': weighted_mse_loss(x, labels, self.loss_weights)}
        if mode == 'predict':
            # 返回所有需要用于评估和可视化的内容
            return x, history, future, history_rkt, metadata
        else:
            raise ValueError(f"Unknown mode: {mode}")