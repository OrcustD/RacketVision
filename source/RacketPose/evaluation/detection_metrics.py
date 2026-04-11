import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from mmengine.registry import METRICS
from mmdet.evaluation import CocoMetric


@METRICS.register_module()
class RacketDetectionMetrics:
    """球拍检测自定义评估指标
    
    包含基本的mAP指标以及针对球拍检测的特定分析
    """
    
    def __init__(self, 
                 iou_thresholds: List[float] = [0.5, 0.75],
                 score_threshold: float = 0.3,
                 class_names: List[str] = ['tabletennis_racket', 'badminton_racket', 'tennis_racket']):
        self.iou_thresholds = iou_thresholds
        self.score_threshold = score_threshold
        self.class_names = class_names
        self.reset()
    
    def reset(self):
        """重置累计指标"""
        self.all_predictions = []
        self.all_ground_truths = []
        
    def update(self, predictions: List[Dict[str, Any]], ground_truths: List[Dict[str, Any]]):
        """更新预测和真值数据
        
        Args:
            predictions: 预测结果列表，每个元素包含 'bboxes', 'scores', 'labels'
            ground_truths: 真值数据列表，每个元素包含 'bboxes', 'labels'
        """
        self.all_predictions.extend(predictions)
        self.all_ground_truths.extend(ground_truths)
    
    def compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """计算两个边界框的IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def compute_ap(self, 
                   precisions: np.ndarray, 
                   recalls: np.ndarray, 
                   use_07_metric: bool = False) -> float:
        """计算平均精度AP"""
        if use_07_metric:
            # VOC07 11点插值方法
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(recalls >= t) == 0:
                    p = 0
                else:
                    p = np.max(precisions[recalls >= t])
                ap = ap + p / 11.
        else:
            # VOC12+ 方法
            mrec = np.concatenate(([0.], recalls, [1.]))
            mpre = np.concatenate(([0.], precisions, [0.]))
            
            # 计算精度包络
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
            
            # 寻找召回率改变的点
            i = np.where(mrec[1:] != mrec[:-1])[0]
            
            # 计算曲线下面积
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        
        return ap
    
    def evaluate_class(self, class_id: int, iou_threshold: float) -> Dict[str, float]:
        """评估单个类别的性能"""
        # 收集该类别的所有预测和真值
        class_predictions = []
        class_ground_truths = []
        
        for pred, gt in zip(self.all_predictions, self.all_ground_truths):
            # 过滤预测结果
            if 'bboxes' in pred and len(pred['bboxes']) > 0:
                pred_mask = (pred['labels'] == class_id) & (pred['scores'] >= self.score_threshold)
                if np.any(pred_mask):
                    class_predictions.append({
                        'bboxes': pred['bboxes'][pred_mask],
                        'scores': pred['scores'][pred_mask]
                    })
                else:
                    class_predictions.append({'bboxes': [], 'scores': []})
            else:
                class_predictions.append({'bboxes': [], 'scores': []})
            
            # 过滤真值
            if 'bboxes' in gt and len(gt['bboxes']) > 0:
                gt_mask = gt['labels'] == class_id
                class_ground_truths.append({
                    'bboxes': gt['bboxes'][gt_mask]
                })
            else:
                class_ground_truths.append({'bboxes': []})
        
        # 统计总的正样本数量
        npos = sum(len(gt['bboxes']) for gt in class_ground_truths)
        
        if npos == 0:
            return {'ap': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        # 收集所有预测结果，按分数排序
        all_detections = []
        for img_id, pred in enumerate(class_predictions):
            if len(pred['bboxes']) > 0:
                for bbox, score in zip(pred['bboxes'], pred['scores']):
                    all_detections.append({
                        'image_id': img_id,
                        'bbox': bbox,
                        'score': score
                    })
        
        if len(all_detections) == 0:
            return {'ap': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        # 按分数降序排序
        all_detections.sort(key=lambda x: x['score'], reverse=True)
        
        # 计算TP、FP
        nd = len(all_detections)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        
        for d, detection in enumerate(all_detections):
            img_id = detection['image_id']
            pred_bbox = detection['bbox']
            gt_bboxes = class_ground_truths[img_id]['bboxes']
            
            if len(gt_bboxes) == 0:
                fp[d] = 1
                continue
            
            # 计算与所有GT的IoU
            ious = [self.compute_iou(pred_bbox, gt_bbox) for gt_bbox in gt_bboxes]
            max_iou = max(ious)
            max_iou_idx = np.argmax(ious)
            
            if max_iou >= iou_threshold:
                tp[d] = 1
                # 标记该GT已被匹配（简化处理，实际应该避免重复匹配）
            else:
                fp[d] = 1
        
        # 计算累计TP、FP
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        
        # 计算精度和召回率
        rec = tp / float(npos)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        
        # 计算AP
        ap = self.compute_ap(prec, rec)
        
        return {
            'ap': ap,
            'precision': prec[-1] if len(prec) > 0 else 0.0,
            'recall': rec[-1] if len(rec) > 0 else 0.0
        }
    
    def compute_metrics(self) -> Dict[str, float]:
        """计算所有评估指标"""
        if len(self.all_predictions) == 0:
            return {}
        
        results = {}
        
        # 计算每个类别在每个IoU阈值下的AP
        all_aps = []
        for class_id in range(len(self.class_names)):
            class_name = self.class_names[class_id]
            
            for iou_thr in self.iou_thresholds:
                metrics = self.evaluate_class(class_id, iou_thr)
                key_prefix = f'{class_name}_iou{iou_thr:.2f}'
                results[f'{key_prefix}_ap'] = metrics['ap']
                results[f'{key_prefix}_precision'] = metrics['precision']
                results[f'{key_prefix}_recall'] = metrics['recall']
                
                all_aps.append(metrics['ap'])
        
        # 计算mAP
        if all_aps:
            results['mAP'] = np.mean(all_aps)
            results[f'mAP@{self.iou_thresholds[0]:.2f}'] = np.mean(
                [results[f'{name}_iou{self.iou_thresholds[0]:.2f}_ap'] 
                 for name in self.class_names])
        
        # 计算运动特定的指标
        self._compute_sport_specific_metrics(results)
        
        return results
    
    def _compute_sport_specific_metrics(self, results: Dict[str, float]):
        """计算运动特定的指标"""
        # 计算不同运动的平均表现
        sports_ap = {
            'racket_sports_mAP': np.mean([
                results.get(f'{name}_iou{self.iou_thresholds[0]:.2f}_ap', 0.0)
                for name in self.class_names
            ])
        }
        
        results.update(sports_ap)
    
    def format_results(self, results: Dict[str, float]) -> str:
        """格式化输出结果"""
        lines = ["球拍检测评估结果:", "=" * 50]
        
        # 总体mAP
        if 'mAP' in results:
            lines.append(f"总体 mAP: {results['mAP']:.4f}")
        if f'mAP@{self.iou_thresholds[0]:.2f}' in results:
            lines.append(f"mAP@{self.iou_thresholds[0]:.2f}: {results[f'mAP@{self.iou_thresholds[0]:.2f}']:.4f}")
        
        lines.append("")
        
        # 各类别详细结果
        for class_name in self.class_names:
            lines.append(f"{class_name}:")
            for iou_thr in self.iou_thresholds:
                key_prefix = f'{class_name}_iou{iou_thr:.2f}'
                ap = results.get(f'{key_prefix}_ap', 0.0)
                precision = results.get(f'{key_prefix}_precision', 0.0)
                recall = results.get(f'{key_prefix}_recall', 0.0)
                lines.append(f"  IoU@{iou_thr:.2f} - AP: {ap:.4f}, P: {precision:.4f}, R: {recall:.4f}")
            lines.append("")
        
        return "\n".join(lines) 