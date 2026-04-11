import numpy as np
from mmengine.registry import HOOKS
from mmengine.hooks import Hook
import cv2
from utils.general import to_img_format, array_to_img
import os
import copy


@HOOKS.register_module()
class HeatmapVisualizerHook(Hook):
    def __init__(self, log_dir: str, save: bool = True, only_batch0: bool = True,
                 add_to_tb: bool = False, target: str = 'ball', ball_bbox_size=5):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.save = save
        self.only_batch0 = only_batch0
        self.add_to_tb = add_to_tb
        self.target = target
        self.ball_bbox_size = ball_bbox_size

    @staticmethod
    def _draw_bbox(image, bbox, color=(0, 255, 0), thickness=2):
        x, y, w, h = np.array(bbox).astype(int).tolist()
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
        return image

    @staticmethod
    def _draw_heatmap(image, heatmap):
        heatmap_ = (copy.deepcopy(heatmap) * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_, cv2.COLORMAP_PARULA)
        overlay = cv2.addWeighted(image, 0.4, heatmap_colored, 0.6, 0)
        return overlay

    @staticmethod
    def _draw_point(image, point, color=(0, 0, 255), radius=2):
        return cv2.circle(image, point, radius=radius, color=color, thickness=-1)

    def _visualize_ball(self, frame, pred, label, coor):
        gt = copy.deepcopy(frame)
        pred_vis = copy.deepcopy(frame)
        gt = self._draw_heatmap(gt, label)
        pred_vis = self._draw_heatmap(pred_vis, pred)
        vis = coor[0] * coor[1]
        if vis:
            gt = self._draw_point(gt, (int(coor[0]), int(coor[1])))
            bbox = [coor[0] - self.ball_bbox_size / 2, coor[1] - self.ball_bbox_size / 2,
                    self.ball_bbox_size, self.ball_bbox_size]
            gt = self._draw_bbox(gt, bbox)
        img = cv2.vconcat([gt, pred_vis])
        return img

    def _visualize_heatmap_by_iter(self, runner, batch_idx, data_batch, outputs, iter_type='val'):
        N, _, H, W = data_batch['frames'].shape
        i = np.random.randint(N)
        data_idx = data_batch['data_idx'][i]
        frames = data_batch['frames'][i].detach().cpu().numpy()
        frame = array_to_img(to_img_format(frames, 3, H, W)[-1])
        img = None

        y_preds, labels, coors = outputs
        pred = y_preds[i][0].detach().cpu().numpy()
        label = labels[i][0].detach().cpu().numpy()
        coor = coors[i].detach().cpu().numpy()
        if self.target == 'ball':
            coor = coor[0]
            img = self._visualize_ball(frame, pred, label, coor)
        else:
            raise ValueError(f'Invalid visualize target: {self.target}')

        if img is not None:
            name = f'{iter_type}_epoch-{runner.epoch}_match-{data_idx[-1][0]}_fid-{data_idx[-1][1]}'
            if self.save:
                cv2.imwrite(f'{self.log_dir}/{name}.jpg', img)
            if self.add_to_tb and runner.visualizer is not None:
                runner.visualizer.add_image(f'vis_{iter_type}', cv2.cvtColor(img, cv2.COLOR_BGR2RGB), step=runner.epoch)

    def after_val_iter(self, runner, batch_idx, data_batch, outputs):
        if batch_idx != 0 and self.only_batch0:
            return
        self._visualize_heatmap_by_iter(runner, batch_idx, data_batch, outputs, iter_type='val')

    def after_test_iter(self, runner, batch_idx, data_batch, outputs):
        if batch_idx != 0 and self.only_batch0:
            return
        self._visualize_heatmap_by_iter(runner, batch_idx, data_batch, outputs, iter_type='test')
