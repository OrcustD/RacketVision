import numpy as np
from mmengine.registry import HOOKS
from mmengine.hooks import Hook
import cv2
import os
import torch


@HOOKS.register_module(name='TrajVis')
class TrajVis(Hook):
    def __init__(self, log_dir: str, only_batch0: bool = True, data_dir: str = 'data',
                 width: int = 1920, height: int = 1080, visualize_racket: bool = False, max_vis: int = 50):
        self.log_dir = log_dir
        self.data_dir = data_dir
        self.width = width
        self.height = height
        self.visualize_racket = visualize_racket
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.only_batch0 = only_batch0
        self.max_vis = max_vis

    def _fetch_background(self, frame_path):
        img = cv2.imread(frame_path)
        if img is None:
            print(f"Warning: Could not read image at {frame_path}")
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return img

    def _visualize_traj(self, frame, history, future, pred):
        """Draw history, GT future, and predicted future on `frame` (BGR, pixel coords)."""
        color_history = (255, 255, 255)
        color_gt = (0, 255, 0)
        color_pred = (0, 0, 255)
        thickness = 5

        history_pts = history.astype(np.int32)
        future_pts = future.astype(np.int32)
        pred_pts = pred.astype(np.int32)

        if history_pts.shape[0] > 1:
            cv2.polylines(frame, [history_pts], isClosed=False, color=color_history,
                          thickness=thickness, lineType=cv2.LINE_AA)
            cv2.circle(frame, tuple(history_pts[-1]), radius=10, color=color_history, thickness=-1)

        if history_pts.shape[0] > 0:
            last_hist_pt = history_pts[-1]

            if future_pts.shape[0] > 0:
                gt_path_to_draw = np.vstack([last_hist_pt, future_pts])
                cv2.polylines(frame, [gt_path_to_draw], isClosed=False, color=color_gt,
                              thickness=thickness, lineType=cv2.LINE_AA)
                cv2.circle(frame, tuple(future_pts[-1]), radius=10, color=color_gt, thickness=-1)

            if pred_pts.shape[0] > 0:
                pred_path_to_draw = np.vstack([last_hist_pt, pred_pts])
                cv2.polylines(frame, [pred_path_to_draw], isClosed=False, color=color_pred,
                              thickness=thickness, lineType=cv2.LINE_AA)
                cv2.circle(frame, tuple(pred_pts[-1]), radius=10, color=color_pred, thickness=-1)

        legend_x = frame.shape[1] - 300
        legend_y = 50
        legend_spacing = 40
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2

        legend_bg = frame[legend_y - 20:legend_y + 160, legend_x - 20:legend_x + 280].copy()
        overlay = np.zeros_like(legend_bg)
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1], overlay.shape[0]), (0, 0, 0), -1)
        legend_bg = cv2.addWeighted(legend_bg, 0.7, overlay, 0.3, 0)
        frame[legend_y - 20:legend_y + 160, legend_x - 20:legend_x + 280] = legend_bg

        cv2.line(frame, (legend_x, legend_y), (legend_x + 30, legend_y), color_history, thickness)
        cv2.circle(frame, (legend_x + 35, legend_y), 5, color_history, -1)
        cv2.putText(frame, "History", (legend_x + 50, legend_y + 5), font, font_scale, color_history, font_thickness)

        cv2.line(frame, (legend_x, legend_y + legend_spacing), (legend_x + 30, legend_y + legend_spacing), color_gt, thickness)
        cv2.circle(frame, (legend_x + 35, legend_y + legend_spacing), 5, color_gt, -1)
        cv2.putText(frame, "Ground Truth", (legend_x + 50, legend_y + legend_spacing + 5), font, font_scale, color_gt, font_thickness)

        cv2.line(frame, (legend_x, legend_y + 2 * legend_spacing), (legend_x + 30, legend_y + 2 * legend_spacing), color_pred, thickness)
        cv2.circle(frame, (legend_x + 35, legend_y + 2 * legend_spacing), 5, color_pred, -1)
        cv2.putText(frame, "Prediction", (legend_x + 50, legend_y + 2 * legend_spacing + 5), font, font_scale, color_pred, font_thickness)

    def _visualize_by_iter(self, runner, batch_idx, data_batch, outputs):
        predictions, history, ground_truth, history_rkt, metadata = outputs
        n, lh, _ = history.shape

        predictions = predictions * torch.tensor([self.width, self.height], device=predictions.device)
        ground_truth = ground_truth * torch.tensor([self.width, self.height], device=ground_truth.device)
        history = history * torch.tensor([self.width, self.height], device=history.device)

        sports = metadata['sport']
        matchs = metadata['match']
        sequences = metadata['sequence']
        start_frames = metadata['start_frame']

        for i in range(n):
            if i >= self.max_vis:
                break
            frame_id = int(start_frames[i]) - 1
            frame_path_start = os.path.join(
                self.data_dir, sports[i], 'all', matchs[i], 'frame', sequences[i], f"{frame_id:04d}.jpg")
            frame_id += lh
            frame_path_middle = os.path.join(
                self.data_dir, sports[i], 'all', matchs[i], 'frame', sequences[i], f"{frame_id:04d}.jpg")
            output_path = os.path.join(self.log_dir, f'{sports[i]}_{matchs[i]}_{sequences[i]}_{frame_id:04d}.jpg')
            frame_1 = self._fetch_background(frame_path_start)
            frame_2 = self._fetch_background(frame_path_middle)
            frame = cv2.addWeighted(frame_1, 0.5, frame_2, 0.5, 0)

            history_np = history[i].cpu().numpy()
            future_np = ground_truth[i].cpu().numpy()
            pred_np = predictions[i].cpu().numpy()

            self._visualize_traj(frame, history_np, future_np, pred_np)

            cv2.imwrite(output_path, frame)

    def after_test_iter(self, runner, batch_idx, data_batch, outputs):
        if batch_idx != 0 and self.only_batch0:
            return
        self._visualize_by_iter(runner, batch_idx, data_batch, outputs)
