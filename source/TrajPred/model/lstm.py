import torch
import torch.nn as nn
from model.loss import weighted_mse_loss
from mmengine.registry import MODELS
from mmengine.model import BaseModel

@MODELS.register_module(name='TPLSTM')
class TrajectoryPredictorLSTM(BaseModel):
    """
    一个基于LSTM的轨迹预测模型，支持两种模式：
    1. 'ball_only': 只使用球的历史轨迹进行预测。
    2. 'ball_racket': 同时使用球和球拍的历史信息进行预测。
    """
    def __init__(self, 
                 mode='ball_only',
                 pred_len=20,
                 # Ball embedding dimensions
                 ball_input_dim=2,
                 ball_embed_dim=32,
                 # Racket embedding dimensions
                 racket_input_dim=10,
                 racket_embed_dim=64,
                 # LSTM parameters
                 lstm_hidden_size=128,
                 lstm_num_layers=2,
                 lstm_dropout=0.1,
                 min_loss_weight=0.5):
        """
        初始化模型.
        
        Args:
            mode (str): 模型模式, 'ball_only' 或 'ball_racket'.
            pred_len (int): 需要预测的未来轨迹长度.
            ball_input_dim (int): 球坐标输入的维度 (默认为2, 即X, Y).
            ball_embed_dim (int): 球坐标嵌入后的维度.
            racket_input_dim (int): 球拍姿态输入的维度 (默认为10, 即5个关键点的X, Y).
            racket_embed_dim (int): 球拍姿态嵌入后的维度.
            lstm_hidden_size (int): LSTM的隐藏层大小.
            lstm_num_layers (int): LSTM的层数.
            lstm_dropout (float): LSTM层之间的dropout概率.
            min_loss_weight (float): 最小的损失权重, 用于加权MSE损失.
        """
        super(TrajectoryPredictorLSTM, self).__init__()
        
        if mode not in ['ball_only', 'ball_racket']:
            raise ValueError("Mode must be 'ball_only' or 'ball_racket'")
        
        self.mode = mode
        self.pred_len = pred_len
        self.loss_weights = torch.linspace(1.0, min_loss_weight, steps=pred_len)

        # --- 1. 定义嵌入层 ---
        if self.mode == 'ball_only':
            # 在ball_only模式下，直接将球坐标嵌入到LSTM的隐藏维度
            self.ball_embed = nn.Linear(ball_input_dim, lstm_hidden_size)
            lstm_input_size = lstm_hidden_size
        else: # ball_racket mode
            # 使用独立的嵌入层处理球和球拍
            self.ball_embed = nn.Linear(ball_input_dim, ball_embed_dim)
            self.racket_embed = nn.Linear(racket_input_dim, racket_embed_dim)
            # LSTM的输入维度是融合后特征的维度
            lstm_input_size = ball_embed_dim + racket_embed_dim

        # --- 2. 定义LSTM核心 ---
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True, # 输入和输出张量将以 (batch, seq, feature) 的形式提供
            dropout=lstm_dropout if lstm_num_layers > 1 else 0
        )
        
        # --- 3. 定义输出解码器 ---
        # 将LSTM的最终隐藏状态映射到预测的轨迹
        self.decoder = nn.Linear(lstm_hidden_size, self.pred_len * 2)

    def _forward(self, ball_hist, racket_hist=None):
        """
        前向传播函数.
        
        Args:
            ball_hist (torch.Tensor): 球的历史轨迹, shape (batch_size, hist_len, 2).
            racket_hist (torch.Tensor, optional): 球拍的历史姿态, shape (batch_size, hist_len, 10).
                                                  在 'ball_racket' 模式下是必需的.
        
        Returns:
            torch.Tensor: 预测的未来轨迹, shape (batch_size, pred_len, 2).
        """
        # --- 1. 特征嵌入和融合 ---
        if self.mode == 'ball_only':
            # (batch_size, hist_len, 2) -> (batch_size, hist_len, lstm_hidden_size)
            embedded_input = self.ball_embed(ball_hist)
        else: # ball_racket mode
            if racket_hist is None:
                raise ValueError("Racket history must be provided in 'ball_racket' mode.")
            # (batch_size, hist_len, 2) -> (batch_size, hist_len, ball_embed_dim)
            embedded_ball = self.ball_embed(ball_hist)
            # (batch_size, hist_len, 10) -> (batch_size, hist_len, racket_embed_dim)
            embedded_racket = self.racket_embed(racket_hist)
            
            # (batch_size, hist_len, ball_embed_dim + racket_embed_dim)
            embedded_input = torch.cat([embedded_ball, embedded_racket], dim=-1)

        # --- 2. 通过LSTM处理序列 ---
        # embedded_input: (batch_size, hist_len, lstm_input_size)
        # lstm_out: (batch_size, hist_len, lstm_hidden_size)
        # (h_n, c_n): 元组, 每个元素shape (num_layers, batch_size, lstm_hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(embedded_input)
        
        # --- 3. 解码预测未来轨迹 ---
        # 我们使用最后一个时间步的LSTM输出来进行预测
        # last_time_step_out: (batch_size, lstm_hidden_size)
        last_time_step_out = lstm_out[:, -1, :]
        
        # (batch_size, lstm_hidden_size) -> (batch_size, pred_len * 2)
        prediction_flat = self.decoder(last_time_step_out)
        
        # (batch_size, pred_len * 2) -> (batch_size, pred_len, 2)
        prediction = prediction_flat.view(-1, self.pred_len, 2)
        
        return prediction

    def forward(self, history=None, future=None, history_rkt=None, future_rkt=None, metadata=None, mode='predict'):
        x = self._forward(history, history_rkt)
        labels = future
        if mode == 'loss':
            return {'loss': weighted_mse_loss(x, labels, self.loss_weights)}
        if mode == 'predict':
            return x, history, future, history_rkt, metadata
        else:
            raise ValueError(f"Unknown mode: {mode}")
