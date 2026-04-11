import torch
import torch.nn as nn
from model.loss import weighted_mse_loss
from mmengine.registry import MODELS
from mmengine.model import BaseModel


@MODELS.register_module(name='TPLSTM')
class TrajectoryPredictorLSTM(BaseModel):
    """LSTM baseline: `ball_only` or concatenated `ball_racket` embeddings."""

    def __init__(self,
                 mode='ball_only',
                 pred_len=20,
                 ball_input_dim=2,
                 ball_embed_dim=32,
                 racket_input_dim=10,
                 racket_embed_dim=64,
                 lstm_hidden_size=128,
                 lstm_num_layers=2,
                 lstm_dropout=0.1,
                 min_loss_weight=0.5):
        super().__init__()

        if mode not in ['ball_only', 'ball_racket']:
            raise ValueError("Mode must be 'ball_only' or 'ball_racket'")

        self.mode = mode
        self.pred_len = pred_len
        self.loss_weights = torch.linspace(1.0, min_loss_weight, steps=pred_len)

        if self.mode == 'ball_only':
            self.ball_embed = nn.Linear(ball_input_dim, lstm_hidden_size)
            lstm_input_size = lstm_hidden_size
        else:
            self.ball_embed = nn.Linear(ball_input_dim, ball_embed_dim)
            self.racket_embed = nn.Linear(racket_input_dim, racket_embed_dim)
            lstm_input_size = ball_embed_dim + racket_embed_dim

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0
        )

        self.decoder = nn.Linear(lstm_hidden_size, self.pred_len * 2)

    def _forward(self, ball_hist, racket_hist=None):
        if self.mode == 'ball_only':
            embedded_input = self.ball_embed(ball_hist)
        else:
            if racket_hist is None:
                raise ValueError("Racket history must be provided in 'ball_racket' mode.")
            embedded_ball = self.ball_embed(ball_hist)
            embedded_racket = self.racket_embed(racket_hist)
            embedded_input = torch.cat([embedded_ball, embedded_racket], dim=-1)

        lstm_out, _ = self.lstm(embedded_input)
        last_time_step_out = lstm_out[:, -1, :]
        prediction_flat = self.decoder(last_time_step_out)
        prediction = prediction_flat.view(-1, self.pred_len, 2)
        return prediction

    def forward(self, history=None, future=None, history_rkt=None, future_rkt=None, metadata=None, mode='predict'):
        x = self._forward(history, history_rkt)
        labels = future
        if mode == 'loss':
            return {'loss': weighted_mse_loss(x, labels, self.loss_weights)}
        if mode == 'predict':
            return x, history, future, history_rkt, metadata
        raise ValueError(f"Unknown mode: {mode}")
