import cv2
import torch
import math
import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS

pred_types = ['TP', 'TN', 'FP1', 'FP2', 'FN']
pred_types_map = {pred_type: i for i, pred_type in enumerate(pred_types)}


@METRICS.register_module(name='BallMetrics')
class BallMetrics(BaseMetric):
    default_prefix = 'Ball'

    def __init__(self, height=288, width=512, tolerance=4, threshold=0.5,
                 last_only=False, gt_src='heatmap', multitask=False):
        super(BallMetrics, self).__init__()
        self.height = height
        self.width = width
        self.tolerance = tolerance
        self.threshold = threshold
        self.last_only = last_only
        self.gt_src = gt_src
        self.multitask = multitask

    def process(self, data_batch, data_samples):
        if self.multitask:
            score, gt, coor, vis, _, _, _, _ = data_samples
        else:
            score, gt, coor = data_samples
        i = data_batch['data_idx']
        if self.last_only:
            i = i[:, -1:, :]
        pred_dict = self._evaluate(i, y_true=gt, y_pred=score, c_true=coor)
        dist_sum = sum(pred_dict['Dist'])
        confusion_matrix = self.get_eval_res(pred_dict)

        self.results.append({
            'TP': confusion_matrix[0],
            'TN': confusion_matrix[1],
            'FP1': confusion_matrix[2],
            'FP2': confusion_matrix[3],
            'FN': confusion_matrix[4],
            'Dist': dist_sum
        })

    def compute_metrics(self, results):
        TP = sum([item['TP'] for item in results])
        TN = sum([item['TN'] for item in results])
        FP1 = sum([item['FP1'] for item in results])
        FP2 = sum([item['FP2'] for item in results])
        FN = sum([item['FN'] for item in results])
        dist = sum([item['Dist'] for item in results])
        accuracy, precision, recall, f1, miss_rate, mean_dist = self.get_metric(TP, TN, FP1, FP2, FN, dist)
        return dict(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            miss_rate=miss_rate,
            mean_dist=mean_dist
        )

    def get_metric(self, TP, TN, FP1, FP2, FN, dist):
        gt_true = TP + FN + FP1
        gt_false = TN + FP2
        pred_true = TP + FP1 + FP2

        accuracy = (TP + TN) / (gt_true + gt_false) if (gt_true + gt_false) > 0 else 0
        precision = TP / pred_true if pred_true > 0 else 0
        recall = TP / gt_true if gt_true > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        miss_rate = 1 - recall
        mean_dist = dist / (TP + FP1) if (TP + FP1) > 0 else 10000

        return accuracy, precision, recall, f1, miss_rate, mean_dist

    def _evaluate(self, indices, y_true=None, y_pred=None, c_true=None, c_pred=None,
                  img_scaler=(1, 1), output_bbox=False, output_gt=False):
        tolerance = self.tolerance
        pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': [], 'Type': [],
                     'BBox': [], 'Confidence': [], 'X_GT': [], 'Y_GT': [],
                     'Visibility_GT': [], 'Dist': []}

        batch_size, seq_len = indices.shape[0], indices.shape[1]
        indices = indices.detach().cpu().numpy().tolist() if torch.is_tensor(indices) else indices.numpy().tolist()

        y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        y_pred = self.to_img_format(y_pred)
        h_pred = y_pred > self.threshold
        pred_len = y_pred.shape[1]

        gt_type = self.gt_src
        if gt_type == 'heatmap' and y_true is not None:
            y_true = y_true.detach().cpu().numpy() if torch.is_tensor(y_true) else y_true
            y_true = self.to_img_format(y_true)
        elif gt_type == 'position' and c_true is not None:
            c_true = c_true.detach().cpu().numpy() if torch.is_tensor(c_true) else c_true
        else:
            raise ValueError('Either y_true or c_true must be provided.')

        for n in range(batch_size):
            prev_d_i = [-1, -1]
            for f in range(seq_len):
                if f >= pred_len:
                    break
                d_i = indices[n][f]
                if d_i != prev_d_i:
                    dist = 0
                    y_p = y_pred[n][f]
                    h_p = h_pred[n][f]
                    bbox_pred = self.predict_location(self.to_img(h_p))
                    cx_pred, cy_pred = int(bbox_pred[0] + bbox_pred[2] / 2), int(bbox_pred[1] + bbox_pred[3] / 2)
                    if np.amax(bbox_pred) > 0:
                        conf = np.amax(y_p[bbox_pred[1]:bbox_pred[1] + bbox_pred[3], bbox_pred[0]:bbox_pred[0] + bbox_pred[2]])
                    else:
                        conf = 0.

                    if gt_type == 'heatmap':
                        y_t = y_true[n][f]
                        bbox_true = self.predict_location(self.to_img(y_t))
                        cx_true, cy_true = int(bbox_true[0] + bbox_true[2] / 2), int(bbox_true[1] + bbox_true[3] / 2)
                    if gt_type == 'position':
                        cx_true, cy_true = c_true[n][f]

                    vis_pred = 0 if cx_pred == 0 and cy_pred == 0 else 1
                    vis_true = 0 if cx_true == 0 and cy_true == 0 else 1

                    if vis_pred == 0 and vis_true == 0:
                        pred_dict['Type'].append(pred_types_map['TN'])
                    elif vis_pred > 0 and vis_true == 0:
                        pred_dict['Type'].append(pred_types_map['FP2'])
                    elif vis_pred == 0 and vis_true > 0:
                        pred_dict['Type'].append(pred_types_map['FN'])
                    elif vis_pred > 0 and vis_true > 0:
                        dist = math.sqrt(pow(cx_pred - cx_true, 2) + pow(cy_pred - cy_true, 2))
                        if dist > tolerance:
                            pred_dict['Type'].append(pred_types_map['FP1'])
                        else:
                            pred_dict['Type'].append(pred_types_map['TP'])
                    else:
                        raise ValueError('Invalid input')

                    pred_dict['Frame'].append(int(d_i[1]))
                    pred_dict['X'].append(int(cx_pred * img_scaler[0]))
                    pred_dict['Y'].append(int(cy_pred * img_scaler[1]))
                    pred_dict['Visibility'].append(vis_pred)
                    pred_dict['Dist'].append(dist)

                    if output_bbox:
                        pred_dict['BBox'].append([int(bbox_pred[0] * img_scaler[0]), int(bbox_pred[1] * img_scaler[1]),
                                                  int(bbox_pred[2] * img_scaler[0]), int(bbox_pred[3] * img_scaler[1])])
                        pred_dict['Confidence'].append(float(conf))

                    if output_gt:
                        vis_gt = 0 if cx_true == 0 and cy_true == 0 else 1
                        pred_dict['X_GT'].append(int(cx_true * img_scaler[0]))
                        pred_dict['Y_GT'].append(int(cy_true * img_scaler[1]))
                        pred_dict['Visibility_GT'].append(vis_gt)

                    prev_d_i = d_i
                else:
                    break

        if not output_bbox:
            del pred_dict['BBox']
            del pred_dict['Confidence']
        if not output_gt:
            del pred_dict['X_GT']
            del pred_dict['Y_GT']
            del pred_dict['Visibility_GT']

        return pred_dict

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

    def get_eval_res(self, pred_dict):
        type_res = np.array(pred_dict['Type'])
        res = np.zeros(5)
        for pred_type in pred_types:
            res[pred_types_map[pred_type]] += int((type_res == pred_types_map[pred_type]).sum())
        return res
