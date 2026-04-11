import torch
import torch.nn as nn
from model.loss import weighted_mse_loss
from mmengine.registry import MODELS
from mmengine.model import BaseModel

@MODELS.register_module(name='TPLSTMAttn')
class TrajectoryPredictorLSTMAttention(BaseModel):
    """
    一个更公平的混合模型：
    1. 使用低维跨注意力融合，且不使用位置编码。
    2. 使用可学习的alpha系数控制融合比例。
    3. 再上投影到高维LSTM进行预测。
    """
    def __init__(self,
                 pred_len=20,
                 ball_input_dim=2,
                 racket_input_dim=10,
                 embed_dim=64,
                 lstm_hidden_size=512,
                 nhead=4,
                 lstm_num_layers=2,
                 dropout=0.1,
                 min_loss_weight=0.5):
        super().__init__()
        
        self.pred_len = pred_len
        self.loss_weights = torch.linspace(1.0, min_loss_weight, steps=pred_len)

        # --- 1. 定义低维嵌入层 ---
        self.ball_embed = nn.Linear(ball_input_dim, embed_dim)
        self.racket_embed = nn.Linear(racket_input_dim, embed_dim)
        
        # --- 2. 定义低维跨注意力融合模块 ---
        # 不再需要 PositionalEncoding
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=nhead,
            dropout=dropout, 
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # --- 3. 定义可学习的融合系数 alpha ---
        # 初始化为0，让模型从最稳定的状态开始学习
        self.alpha = nn.Parameter(torch.tensor(0.0))
        
        # --- 4. 定义“上投影层” ---
        self.fusion_to_lstm_projection = nn.Linear(embed_dim, lstm_hidden_size)
        
        # --- 5. 定义高维LSTM核心模块 ---
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0
        )
        
        # --- 6. 定义输出解码器 ---
        self.decoder = nn.Linear(lstm_hidden_size, self.pred_len * 2)

    def _forward(self, ball_hist, racket_hist):
        # 1. 低维嵌入
        ball_seq = self.ball_embed(ball_hist)
        racket_seq = self.racket_embed(racket_hist)
        
        # 2. 在低维空间进行跨注意力融合 (无位置编码)
        attended_ball_features, _ = self.cross_attention(
            query=ball_seq, key=racket_seq, value=racket_seq)
        
        # 3. 使用alpha进行门控残差连接和层归一化
        # ball_seq是主干信息，attended_ball_features是融合进来的辅助信息
        fused_seq_low_dim = self.norm1(ball_seq + self.alpha * self.dropout1(attended_ball_features))
        
        # 4. 上投影到高维空间
        fused_seq_high_dim = self.fusion_to_lstm_projection(fused_seq_low_dim)
        
        # 5. 通过LSTM进行时序建模
        lstm_output, _ = self.lstm(fused_seq_high_dim)
        
        # 6. 解码
        last_time_step_out = lstm_output[:, -1, :]
        prediction_flat = self.decoder(last_time_step_out)
        prediction = prediction_flat.view(-1, self.pred_len, 2)
        
        return prediction

    # forward 函数与之前完全相同，这里省略
    def forward(self, history=None, future=None, history_rkt=None, future_rkt=None, metadata=None, mode='predict'):
        if history_rkt is None:
            raise ValueError("LSTMAttn model requires racket history ('history_rkt').")
        x = self._forward(history, history_rkt)
        labels = future
        if mode == 'loss':
            # 在训练时可以打印alpha的值来观察它的变化
            if torch.rand(1) < 0.01: # 以1%的概率打印
                print(f"Current alpha: {self.alpha.item():.4f}")
            return {'loss': weighted_mse_loss(x, labels, self.loss_weights)}
        if mode == 'predict':
            return x, history, future, history_rkt, metadata
        else:
            raise ValueError(f"Unknown mode: {mode}")