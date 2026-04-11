import torch
import torch.nn as nn
from model.loss import weighted_mse_loss
from mmengine.registry import MODELS
from mmengine.model import BaseModel


@MODELS.register_module(name='TPLSTMAttn')
class TrajectoryPredictorLSTMAttention(BaseModel):
    """Cross-attention over low-dim embeddings, gated residual, then LSTM head (released model)."""

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

        self.ball_embed = nn.Linear(ball_input_dim, embed_dim)
        self.racket_embed = nn.Linear(racket_input_dim, embed_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.alpha = nn.Parameter(torch.tensor(0.0))

        self.fusion_to_lstm_projection = nn.Linear(embed_dim, lstm_hidden_size)

        self.lstm = nn.LSTM(
            input_size=lstm_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0
        )

        self.decoder = nn.Linear(lstm_hidden_size, self.pred_len * 2)

    def _forward(self, ball_hist, racket_hist):
        ball_seq = self.ball_embed(ball_hist)
        racket_seq = self.racket_embed(racket_hist)

        attended_ball_features, _ = self.cross_attention(
            query=ball_seq, key=racket_seq, value=racket_seq)

        fused_seq_low_dim = self.norm1(
            ball_seq + self.alpha * self.dropout1(attended_ball_features))

        fused_seq_high_dim = self.fusion_to_lstm_projection(fused_seq_low_dim)

        lstm_output, _ = self.lstm(fused_seq_high_dim)

        last_time_step_out = lstm_output[:, -1, :]
        prediction_flat = self.decoder(last_time_step_out)
        prediction = prediction_flat.view(-1, self.pred_len, 2)

        return prediction

    def forward(self, history=None, future=None, history_rkt=None, future_rkt=None, metadata=None, mode='predict'):
        if history_rkt is None:
            raise ValueError("LSTMAttn model requires racket history ('history_rkt').")
        x = self._forward(history, history_rkt)
        labels = future
        if mode == 'loss':
            return {'loss': weighted_mse_loss(x, labels, self.loss_weights)}
        if mode == 'predict':
            return x, history, future, history_rkt, metadata
        raise ValueError(f"Unknown mode: {mode}")
