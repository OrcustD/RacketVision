import cv2
import torch
import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS
from mmdet.evaluation.functional import bbox_overlaps


@METRICS.register_module(name='BallCOCOMetrics')
class BallCOCOMetrics(BaseMetric):
    default_prefix = 'Ball'

    def __init__(self, height=288, width=512, tolerance=4, threshold=0.5,
                 last_only=False, gt_src='heatmap', multitask=False, bbox_size=7):
        super(BallCOCOMetrics, self).__init__()
        self.height = height
        self.width = width
        self.threshold = threshold
        self.last_only = last_only
        self.gt_src = gt_src
        self.multitask = multitask
        self.bbox_size = bbox_size

    def process(self, data_batch, data_samples):
        if self.multitask:
            score, gt, coor, vis, _, _, _, _ = data_samples
        else:
            score, gt, coor = data_samples
        i = data_batch['data_idx']
        if self.last_only:
            i = i[:, -1:, :]
        bbox_preds, bbox_gts = self._evaluate(i, y_true=gt, y_pred=score, c_true=coor)

    def compute_metrics(self, results):
        AP50 = sum([item['ap50'] for item in results])
        AP75 = sum([item['ap75'] for item in results])
        Pos = sum([item['pos'] for item in results])
        map50 = AP50 / Pos if Pos > 0 else 0
        map75 = AP75 / Pos if Pos > 0 else 0
        return dict(
            mAP_50=map50,
            mAP_75=map75,
        )

    @staticmethod
    def cxcy_to_xyxy(cx, cy, bbox_size):
        if cx == 0 and cy == 0:
            return np.array([0, 0, 0, 0])
        xyxy = [int(cx - bbox_size / 2), int(cy - bbox_size / 2),
                int(cx + bbox_size / 2), int(cy + bbox_size / 2)]
        return np.array(xyxy)

    @staticmethod
    def xywh_to_xyxy(bboxes):
        if len(bboxes) == 0:
            return bboxes
        if isinstance(bboxes, list) or isinstance(bboxes, tuple):
            bboxes = np.array(bboxes)
        x, y, w, h = bboxes[0], bboxes[1], bboxes[2], bboxes[3]
        return np.array([x, y, x + w, y + h])

    def _evaluate(self, indices, y_true=None, y_pred=None, c_true=None, c_pred=None,
                  img_scaler=(1, 1), output_bbox=False, output_gt=False):
        batch_size, seq_len = indices.shape[0], indices.shape[1]
        indices = indices.detach().cpu().numpy().tolist() if torch.is_tensor(indices) else indices.numpy().tolist()

        y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        y_pred = self.to_img_format(y_pred)
        h_pred = y_pred > self.threshold
        pred_len = y_pred.shape[1]

        gt_type = self.gt_src
        use_pose_gt = False
        if gt_type == 'heatmap' and y_true is not None:
            y_true = y_true.detach().cpu().numpy() if torch.is_tensor(y_true) else y_true
            y_true = self.to_img_format(y_true)
        elif gt_type == 'position' and c_true is not None:
            use_pose_gt = True
            c_true = c_true.detach().cpu().numpy() if torch.is_tensor(c_true) else c_true
        else:
            raise ValueError('Either y_true or c_true must be provided.')

        bbox_preds = []
        bbox_gts = []

        for n in range(batch_size):
            ap_50 = 0
            ap_75 = 0
            pos = 0
            prev_d_i = [-1, -1]
            for f in range(seq_len):
                if f >= pred_len:
                    break
                d_i = indices[n][f]
                if d_i != prev_d_i:
                    y_p = y_pred[n][f]
                    h_p = h_pred[n][f]
                    h_t = y_true[n][f]
                    bbox_pred = self.xywh_to_xyxy(self.predict_location(self.to_img(h_p))).reshape(1, -1)
                    if use_pose_gt:
                        bbox_gt = self.cxcy_to_xyxy(c_true[n][f][0], c_true[n][f][1], self.bbox_size).reshape(1, -1)
                    else:
                        bbox_gt = self.xywh_to_xyxy(self.predict_location(self.to_img(h_t))).reshape(1, -1)
                    if bbox_gt.sum() > 0:
                        pos += 1
                    iou_matrix = bbox_overlaps(bbox_gt, bbox_pred)
                    iou = iou_matrix[0, 0]
                    if iou >= 0.5:
                        ap_50 += 1
                        if iou >= 0.75:
                            ap_75 += 1
                self.results.append(dict(
                    ap50=ap_50,
                    ap75=ap_75,
                    pos=pos
                ))
            bbox_preds.append(bbox_pred)
            bbox_gts.append(bbox_gt)
        return bbox_preds, bbox_gts

    def to_img(self, image):
        image = image * 255
        image = image.astype('uint8')
        return image

    def to_img_format(self, input, num_ch=1):
        assert len(input.shape) == 4, 'Input must be 4D tensor.'
        if num_ch == 1:
            return input
        else:
            input = np.transpose(input, (0, 2, 3, 1))
            seq_len = int(input.shape[-1] / num_ch)
            img_seq = np.array([]).reshape(0, seq_len, self.height, self.width, 3)
            for n in range(input.shape[0]):
                frame = np.array([]).reshape(0, self.height, self.width, 3)
                for f in range(0, input.shape[-1], num_ch):
                    img = input[n, :, :, f:f + 3]
                    frame = np.concatenate((frame, img.reshape(1, self.height, self.width, 3)), axis=0)
                img_seq = np.concatenate((img_seq, frame.reshape(1, seq_len, self.height, self.width, 3)), axis=0)
            return img_seq

    def predict_location(self, heatmap):
        if np.amax(heatmap) == 0:
            return 0, 0, 0, 0
        else:
            (cnts, _) = cv2.findContours(heatmap.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rects = [cv2.boundingRect(ctr) for ctr in cnts]
            max_area_idx = 0
            max_area = rects[0][2] * rects[0][3]
            for i in range(1, len(rects)):
                area = rects[i][2] * rects[i][3]
                if area > max_area:
                    max_area_idx = i
                    max_area = area
            x, y, w, h = rects[max_area_idx]
            return x, y, w, h
