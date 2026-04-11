import numpy as np
from mmengine.registry import HOOKS
from mmengine.hooks import Hook
import cv2
import os
import torch

@HOOKS.register_module(name='TrajVis')
class TrajVis(Hook):
    def __init__(self, log_dir: str, only_batch0: bool=True, data_dir: str='data', width: int=1920, height: int=1080, visualize_racket: bool=False, max_vis: int=50):
        # self.visualizer = HeatmapVisualizer(log_dir)
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
            """
            在单帧图像上绘制历史、真实和预测的轨迹。

            Args:
                frame (np.ndarray): 从视频中读取的图像帧，用于绘制的背景。
                history (np.ndarray): 历史轨迹点 (N, 2)，像素坐标。
                future (np.ndarray): 真实的未来轨迹点 (M, 2)，像素坐标。
                pred (np.ndarray): 预测的未来轨迹点 (M, 2)，像素坐标。
            """
            # --- 1. 定义颜色和线条粗细 ---
            # OpenCV使用BGR格式，所以红色是(0, 0, 255)
            color_history = (255, 255, 255)  # 白色
            color_gt = (0, 255, 0)          # 绿色
            color_pred = (0, 0, 255)        # 红色
            thickness = 5  # 线条粗细

            # --- 2. 将浮点坐标转换为整数，以便绘图 ---
            history_pts = history.astype(np.int32)
            future_pts = future.astype(np.int32)
            pred_pts = pred.astype(np.int32)
            
            # --- 3. 绘制历史轨迹 ---
            if history_pts.shape[0] > 1:
                # cv2.polylines可以一次性画出所有连接的线段
                cv2.polylines(frame, [history_pts], isClosed=False, color=color_history, 
                            thickness=thickness, lineType=cv2.LINE_AA)
                # 在历史轨迹的最后一个点画一个实心圆
                cv2.circle(frame, tuple(history_pts[-1]), radius=10, color=color_history, thickness=-1)

            # --- 4. 绘制未来轨迹（真实和预测）---
            # 为了视觉上更连贯，我们将未来轨迹的起点连接到历史轨迹的终点
            if history_pts.shape[0] > 0:
                last_hist_pt = history_pts[-1]

                # 绘制真实未来轨迹
                if future_pts.shape[0] > 0:
                    # 将历史终点与未来轨迹点连接起来
                    gt_path_to_draw = np.vstack([last_hist_pt, future_pts])
                    cv2.polylines(frame, [gt_path_to_draw], isClosed=False, color=color_gt, 
                                thickness=thickness, lineType=cv2.LINE_AA)
                    # 在真实轨迹的终点画一个实心圆
                    cv2.circle(frame, tuple(future_pts[-1]), radius=10, color=color_gt, thickness=-1)

                # 绘制预测未来轨迹
                if pred_pts.shape[0] > 0:
                    # 将历史终点与预测轨迹点连接起来
                    pred_path_to_draw = np.vstack([last_hist_pt, pred_pts])
                    cv2.polylines(frame, [pred_path_to_draw], isClosed=False, color=color_pred, 
                                thickness=thickness, lineType=cv2.LINE_AA)
                    # 在预测轨迹的终点画一个实心圆
                    cv2.circle(frame, tuple(pred_pts[-1]), radius=10, color=color_pred, thickness=-1)
            
            # --- 5. 添加图例 ---
            # 图例位置和样式设置
            legend_x = frame.shape[1] - 300  # 距离右边缘300像素
            legend_y = 50                    # 距离顶部50像素
            legend_spacing = 40              # 每行间距
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2
            
            # 绘制半透明背景
            legend_bg = frame[legend_y-20:legend_y+160, legend_x-20:legend_x+280].copy()
            overlay = np.zeros_like(legend_bg)
            cv2.rectangle(overlay, (0, 0), (overlay.shape[1], overlay.shape[0]), (0, 0, 0), -1)
            legend_bg = cv2.addWeighted(legend_bg, 0.7, overlay, 0.3, 0)
            frame[legend_y-20:legend_y+160, legend_x-20:legend_x+280] = legend_bg
            
            # 绘制图例项目
            # 历史轨迹
            cv2.line(frame, (legend_x, legend_y), (legend_x + 30, legend_y), color_history, thickness)
            cv2.circle(frame, (legend_x + 35, legend_y), 5, color_history, -1)
            cv2.putText(frame, "History", (legend_x + 50, legend_y + 5), font, font_scale, color_history, font_thickness)
            
            # 真实轨迹
            cv2.line(frame, (legend_x, legend_y + legend_spacing), (legend_x + 30, legend_y + legend_spacing), color_gt, thickness)
            cv2.circle(frame, (legend_x + 35, legend_y + legend_spacing), 5, color_gt, -1)
            cv2.putText(frame, "Ground Truth", (legend_x + 50, legend_y + legend_spacing + 5), font, font_scale, color_gt, font_thickness)
            
            # 预测轨迹
            cv2.line(frame, (legend_x, legend_y + 2*legend_spacing), (legend_x + 30, legend_y + 2*legend_spacing), color_pred, thickness)
            cv2.circle(frame, (legend_x + 35, legend_y + 2*legend_spacing), 5, color_pred, -1)
            cv2.putText(frame, "Prediction", (legend_x + 50, legend_y + 2*legend_spacing + 5), font, font_scale, color_pred, font_thickness)
    
    def _visualize_by_iter(self, runner, batch_idx, data_batch, outputs):
        predictions, history, ground_truth, history_rkt, metadata = outputs
        N, Lh, _ = history.shape
        
        predictions = predictions * torch.tensor([self.width, self.height], device=predictions.device)
        ground_truth = ground_truth * torch.tensor([self.width, self.height], device=ground_truth.device)
        history = history * torch.tensor([self.width, self.height], device=history.device)
        # history_rkt = history_rkt * torch.tensor([self.width, self.height], device=history_rkt.device)
        
        sports = metadata['sport']
        matchs = metadata['match']
        sequences = metadata['sequence']
        start_frames = metadata['start_frame']

        for i in range(N):
            if i >= self.max_vis:
                break
            frame_id = int(start_frames[i]) -1
            frame_path_start = os.path.join(self.data_dir, sports[i], 'all', matchs[i], 'frame', sequences[i], f"{frame_id:04d}.jpg")
            frame_id += Lh
            frame_path_middle = os.path.join(self.data_dir, sports[i], 'all', matchs[i], 'frame', sequences[i], f"{frame_id:04d}.jpg")
            output_path = os.path.join(self.log_dir, f'{sports[i]}_{matchs[i]}_{sequences[i]}_{frame_id:04d}.jpg')
            frame_1 = self._fetch_background(frame_path_start)
            frame_2 = self._fetch_background(frame_path_middle)
            # Mix frame_1 and frame_2 with 50% opacity each
            frame = cv2.addWeighted(frame_1, 0.5, frame_2, 0.5, 0)
            
            # Visualize the trajectory
            history_np = history[i].cpu().numpy()
            future_np = ground_truth[i].cpu().numpy()
            pred_np = predictions[i].cpu().numpy()
            # racket_pose = (history_rkt[i][-1].reshape(-1, 2) * torch.tensor([self.width, self.height], device=history_rkt.device)).cpu().numpy()

            self._visualize_traj(frame, history_np, future_np, pred_np)

            # Save the visualized frame
            cv2.imwrite(output_path, frame)
        
    def after_test_iter(self, runner, batch_idx, data_batch, outputs):
        if batch_idx != 0 and self.only_batch0:
            return
        self._visualize_by_iter(runner, batch_idx, data_batch, outputs)
