import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from mmengine.registry import METRICS
from mmpose.evaluation import PCKAccuracy, AUC, NME


@METRICS.register_module()
class RacketPoseMetrics:
    """球拍姿态估计自定义评估指标
    
    包含PCK、MPJPE、AUC、NME等指标以及针对球拍关键点的特定分析
    """
    
    def __init__(self, 
                 pck_thresholds: List[float] = [0.1, 0.2, 0.3],
                 keypoint_names: List[str] = ['top', 'bottom', 'handle', 'left', 'right'],
                 normalize_item: str = 'bbox',
                 sigma: List[float] = [0.025, 0.05, 0.025, 0.1, 0.1]):
        self.pck_thresholds = pck_thresholds
        self.keypoint_names = keypoint_names
        self.normalize_item = normalize_item
        self.sigma = sigma  # 用于OKS计算的sigma值
        self.reset()
    
    def reset(self):
        """重置累计指标"""
        self.all_predictions = []
        self.all_ground_truths = []
        self.all_bbox_sizes = []
        
    def update(self, 
               predictions: List[np.ndarray], 
               ground_truths: List[np.ndarray], 
               bbox_sizes: List[float]):
        """更新预测和真值数据
        
        Args:
            predictions: 预测关键点列表，shape为(N, 5, 2)
            ground_truths: 真值关键点列表，shape为(N, 5, 2) 
            bbox_sizes: 对应的边界框尺寸列表，用于标准化
        """
        self.all_predictions.extend(predictions)
        self.all_ground_truths.extend(ground_truths)
        self.all_bbox_sizes.extend(bbox_sizes)
    
    def compute_pck(self, 
                    pred_kpts: np.ndarray, 
                    gt_kpts: np.ndarray, 
                    bbox_size: float, 
                    threshold: float) -> Tuple[np.ndarray, int]:
        """计算PCK指标
        
        Args:
            pred_kpts: 预测关键点 (5, 2)
            gt_kpts: 真值关键点 (5, 2)
            bbox_size: 边界框尺寸
            threshold: PCK阈值
            
        Returns:
            correct: 每个关键点是否正确 (5,)
            visible_count: 可见关键点数量
        """
        # 计算距离
        distances = np.linalg.norm(pred_kpts - gt_kpts, axis=1)
        
        # 标准化距离
        if self.normalize_item == 'bbox':
            normalized_distances = distances / bbox_size
        else:
            # 使用头部尺寸标准化（top到bottom的距离）
            head_size = np.linalg.norm(gt_kpts[0] - gt_kpts[1])
            normalized_distances = distances / max(head_size, 1e-6)
        
        # 判断是否正确（距离小于阈值且关键点可见）
        visible_mask = ~np.isnan(gt_kpts).any(axis=1)
        correct = (normalized_distances < threshold) & visible_mask
        visible_count = np.sum(visible_mask)
        
        return correct, visible_count
    
    def compute_mpjpe(self, pred_kpts: np.ndarray, gt_kpts: np.ndarray) -> float:
        """计算MPJPE (Mean Per Joint Position Error)"""
        # 只计算可见关键点的误差
        visible_mask = ~np.isnan(gt_kpts).any(axis=1)
        if not np.any(visible_mask):
            return float('nan')
        
        distances = np.linalg.norm(pred_kpts - gt_kpts, axis=1)
        return np.mean(distances[visible_mask])
    
    def compute_oks(self, pred_kpts: np.ndarray, gt_kpts: np.ndarray, bbox_area: float) -> float:
        """计算OKS (Object Keypoint Similarity)"""
        visible_mask = ~np.isnan(gt_kpts).any(axis=1)
        if not np.any(visible_mask):
            return 0.0
        
        # 计算距离的平方
        distances_sq = np.sum((pred_kpts - gt_kpts) ** 2, axis=1)
        
        # 计算每个关键点的OKS
        oks_per_kpt = np.zeros(len(self.keypoint_names))
        for i in range(len(self.keypoint_names)):
            if visible_mask[i]:
                s = bbox_area
                sigma_i = self.sigma[i] if i < len(self.sigma) else 0.1
                oks_per_kpt[i] = np.exp(-distances_sq[i] / (2 * s * (sigma_i ** 2)))
        
        # 返回可见关键点的平均OKS
        return np.mean(oks_per_kpt[visible_mask])
    
    def compute_auc(self, pck_scores: List[float], thresholds: List[float]) -> float:
        """计算PCK曲线下面积AUC"""
        if len(pck_scores) < 2:
            return 0.0
        
        # 使用梯形法则计算AUC
        return np.trapz(pck_scores, thresholds)
    
    def compute_nme(self, pred_kpts: np.ndarray, gt_kpts: np.ndarray) -> float:
        """计算归一化平均误差NME"""
        # 使用left和right关键点距离作为标准化因子
        left_pt = gt_kpts[3]   # left keypoint
        right_pt = gt_kpts[4]  # right keypoint
        
        if np.isnan(left_pt).any() or np.isnan(right_pt).any():
            return float('nan')
        
        norm_factor = np.linalg.norm(left_pt - right_pt)
        if norm_factor < 1e-6:
            return float('nan')
        
        # 计算所有可见关键点的平均误差
        visible_mask = ~np.isnan(gt_kpts).any(axis=1)
        if not np.any(visible_mask):
            return float('nan')
        
        distances = np.linalg.norm(pred_kpts - gt_kpts, axis=1)
        mean_distance = np.mean(distances[visible_mask])
        
        return mean_distance / norm_factor
    
    def compute_metrics(self) -> Dict[str, float]:
        """计算所有评估指标"""
        if len(self.all_predictions) == 0:
            return {}
        
        results = {}
        
        # 初始化累计变量
        total_correct = {thr: np.zeros(len(self.keypoint_names)) for thr in self.pck_thresholds}
        total_visible = np.zeros(len(self.keypoint_names))
        
        mpjpe_values = []
        oks_values = []
        nme_values = []
        
        # 逐样本计算指标
        for pred, gt, bbox_size in zip(self.all_predictions, self.all_ground_truths, self.all_bbox_sizes):
            # 确保输入格式正确
            if pred.shape != (len(self.keypoint_names), 2) or gt.shape != (len(self.keypoint_names), 2):
                continue
            
            # 计算PCK
            for thr in self.pck_thresholds:
                correct, visible_count = self.compute_pck(pred, gt, bbox_size, thr)
                total_correct[thr] += correct.astype(float)
            
            # 统计可见关键点
            visible_mask = ~np.isnan(gt).any(axis=1)
            total_visible += visible_mask.astype(float)
            
            # 计算MPJPE
            mpjpe = self.compute_mpjpe(pred, gt)
            if not np.isnan(mpjpe):
                mpjpe_values.append(mpjpe)
            
            # 计算OKS
            bbox_area = bbox_size ** 2
            oks = self.compute_oks(pred, gt, bbox_area)
            oks_values.append(oks)
            
            # 计算NME
            nme = self.compute_nme(pred, gt)
            if not np.isnan(nme):
                nme_values.append(nme)
        
        # 计算PCK指标
        for thr in self.pck_thresholds:
            # 整体PCK
            total_correct_all = np.sum(total_correct[thr])
            total_visible_all = np.sum(total_visible)
            if total_visible_all > 0:
                results[f'PCK@{thr:.1f}'] = total_correct_all / total_visible_all
            
            # 每个关键点的PCK
            for i, kpt_name in enumerate(self.keypoint_names):
                if total_visible[i] > 0:
                    pck_kpt = total_correct[thr][i] / total_visible[i]
                    results[f'PCK@{thr:.1f}_{kpt_name}'] = pck_kpt
        
        # 计算其他指标
        if mpjpe_values:
            results['MPJPE'] = np.mean(mpjpe_values)
        
        if oks_values:
            results['mOKS'] = np.mean(oks_values)
        
        if nme_values:
            results['NME'] = np.mean(nme_values)
        
        # 计算AUC
        if len(self.pck_thresholds) > 1:
            pck_scores = [results.get(f'PCK@{thr:.1f}', 0.0) for thr in self.pck_thresholds]
            results['AUC'] = self.compute_auc(pck_scores, self.pck_thresholds)
        
        # 计算球拍特定指标
        self._compute_racket_specific_metrics(results)
        
        return results
    
    def _compute_racket_specific_metrics(self, results: Dict[str, float]):
        """计算球拍特定的指标"""
        # 计算不同关键点组的性能
        
        # 球拍头部关键点 (top, bottom, left, right)
        head_kpts = ['top', 'bottom', 'left', 'right']
        head_pck_values = []
        for thr in self.pck_thresholds:
            head_pcks = [results.get(f'PCK@{thr:.1f}_{kpt}', 0.0) for kpt in head_kpts]
            if head_pcks:
                head_pck = np.mean(head_pcks)
                results[f'PCK@{thr:.1f}_head'] = head_pck
                if thr == self.pck_thresholds[0]:  # 使用第一个阈值作为主要指标
                    head_pck_values.append(head_pck)
        
        # 关键结构点 (top, bottom, handle)
        key_kpts = ['top', 'bottom', 'handle']
        for thr in self.pck_thresholds:
            key_pcks = [results.get(f'PCK@{thr:.1f}_{kpt}', 0.0) for kpt in key_kpts]
            if key_pcks:
                results[f'PCK@{thr:.1f}_key_structure'] = np.mean(key_pcks)
        
        # 球拍对称性评估 (left vs right)
        for thr in self.pck_thresholds:
            left_pck = results.get(f'PCK@{thr:.1f}_left', 0.0)
            right_pck = results.get(f'PCK@{thr:.1f}_right', 0.0)
            if left_pck > 0 and right_pck > 0:
                symmetry_score = 1.0 - abs(left_pck - right_pck)
                results[f'symmetry_score@{thr:.1f}'] = symmetry_score
    
    def format_results(self, results: Dict[str, float]) -> str:
        """格式化输出结果"""
        lines = ["球拍姿态估计评估结果:", "=" * 50]
        
        # 主要指标
        main_metrics = ['PCK@0.2', 'MPJPE', 'mOKS', 'NME', 'AUC']
        for metric in main_metrics:
            if metric in results:
                lines.append(f"{metric}: {results[metric]:.4f}")
        
        lines.append("")
        
        # PCK详细结果
        lines.append("PCK详细结果:")
        for thr in self.pck_thresholds:
            lines.append(f"  PCK@{thr:.1f}:")
            # 整体PCK
            overall_pck = results.get(f'PCK@{thr:.1f}', 0.0)
            lines.append(f"    整体: {overall_pck:.4f}")
            
            # 各关键点PCK
            for kpt_name in self.keypoint_names:
                kpt_pck = results.get(f'PCK@{thr:.1f}_{kpt_name}', 0.0)
                lines.append(f"    {kpt_name}: {kpt_pck:.4f}")
            
            # 特定组合
            head_pck = results.get(f'PCK@{thr:.1f}_head', 0.0)
            key_pck = results.get(f'PCK@{thr:.1f}_key_structure', 0.0)
            symmetry = results.get(f'symmetry_score@{thr:.1f}', 0.0)
            
            if head_pck > 0:
                lines.append(f"    头部平均: {head_pck:.4f}")
            if key_pck > 0:
                lines.append(f"    关键结构: {key_pck:.4f}")
            if symmetry > 0:
                lines.append(f"    对称性: {symmetry:.4f}")
            
            lines.append("")
        
        return "\n".join(lines) 