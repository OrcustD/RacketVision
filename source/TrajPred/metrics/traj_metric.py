import torch
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS

@METRICS.register_module(name='ADE')
class ADE(BaseMetric):
    """Average Displacement Error Metric.
    
    Calculates the mean of the average L2 distance between predicted points
    and ground truth points over the prediction sequence.
    """
    default_prefix = 'ade'  # 用于在日志中显示指标名称的前缀

    def process(self, data_batch: dict, data_samples: tuple):
        """
        处理一个批次的数据，计算该批次的ADE总和。

        Args:
            data_batch (dict): 模型输入，在这里我们不需要使用它。
            data_samples (tuple): 模型的预测输出。根据您的模型，
                                  它是一个包含 (predictions, history, ground_truth, ...) 的元组。
        """
        # 从模型输出中解包预测值和真实值
        # 形状: (batch_size, pred_len, 2)
        predictions, _, ground_truth, _, _ = data_samples

        # 计算每个时间步的欧氏距离
        # (batch_size, pred_len, 2) -> (batch_size, pred_len)
        width = 1920.0
        height = 1080.0
        predictions = predictions * torch.tensor([width, height], device=predictions.device)
        ground_truth = ground_truth * torch.tensor([width, height], device=ground_truth.device)

        dist = torch.linalg.norm(predictions - ground_truth, dim=-1)
        
        # 计算每个轨迹的平均位移误差 (ade)
        # (batch_size, pred_len) -> (batch_size,)
        ade_per_sample = dist.mean(dim=-1)
        
        # 将当前批次的结果添加到 self.results 列表中
        self.results.append({
            'batch_size': len(ground_truth),
            'ade_sum': ade_per_sample.sum().cpu()
        })

    def compute_metrics(self, results: list) -> dict:
        """
        在所有批次处理完毕后，计算最终的ADE指标。

        Args:
            results (list): 从 process 方法收集的结果列表。
        
        Returns:
            dict: 包含最终ADE值的字典。
        """
        total_ade = sum(item['ade_sum'] for item in results)
        total_samples = sum(item['batch_size'] for item in results)
        
        # 避免除以零的错误
        if total_samples == 0:
            return {self.prefix: 0.0}
            
        final_ade = total_ade / total_samples
        return {self.prefix: final_ade.item()}

@METRICS.register_module(name='FDE')
class FDE(BaseMetric):
    """Final Displacement Error Metric.

    Calculates the mean L2 distance between the final predicted point and
    the final ground truth point.
    """
    default_prefix = 'fde' # 日志前缀

    def process(self, data_batch: dict, data_samples: tuple):
        """
        处理一个批次的数据，计算该批次的FDE总和。
        """
        # 形状: (batch_size, pred_len, 2)
        predictions, _, ground_truth, _, _ = data_samples

        # 只取最后一个时间点的预测和真实值
        # 形状: (batch_size, 2)
        final_pred = predictions[:, -1, :]
        final_gt = ground_truth[:, -1, :]

        width = 1920.0
        height = 1080.0
        final_pred = final_pred * torch.tensor([width, height], device=final_pred.device)
        final_gt = final_gt * torch.tensor([width, height], device=final_gt.device)
        
        # 计算最后一个时间点的欧氏距离
        # (batch_size, 2) -> (batch_size,)
        fde_per_sample = torch.linalg.norm(final_pred - final_gt, dim=-1)

        self.results.append({
            'batch_size': len(ground_truth),
            'fde_sum': fde_per_sample.sum().cpu()
        })

    def compute_metrics(self, results: list) -> dict:
        """
        在所有批次处理完毕后，计算最终的FDE指标。
        """
        total_fde = sum(item['fde_sum'] for item in results)
        total_samples = sum(item['batch_size'] for item in results)

        if total_samples == 0:
            return {self.prefix: 0.0}
            
        final_fde = total_fde / total_samples
        return {self.prefix: final_fde.item()}