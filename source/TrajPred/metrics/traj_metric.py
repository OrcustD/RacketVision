import torch
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS


@METRICS.register_module(name='ADE')
class ADE(BaseMetric):
    """Average displacement error: mean L2 over all predicted timesteps (pixel space after denorm)."""

    default_prefix = 'ade'

    def process(self, data_batch: dict, data_samples: tuple):
        predictions, _, ground_truth, _, _ = data_samples

        width = 1920.0
        height = 1080.0
        predictions = predictions * torch.tensor([width, height], device=predictions.device)
        ground_truth = ground_truth * torch.tensor([width, height], device=ground_truth.device)

        dist = torch.linalg.norm(predictions - ground_truth, dim=-1)
        ade_per_sample = dist.mean(dim=-1)

        self.results.append({
            'batch_size': len(ground_truth),
            'ade_sum': ade_per_sample.sum().cpu()
        })

    def compute_metrics(self, results: list) -> dict:
        total_ade = sum(item['ade_sum'] for item in results)
        total_samples = sum(item['batch_size'] for item in results)

        if total_samples == 0:
            return {self.prefix: 0.0}

        final_ade = total_ade / total_samples
        return {self.prefix: final_ade.item()}


@METRICS.register_module(name='FDE')
class FDE(BaseMetric):
    """Final displacement error: L2 at the last predicted timestep (pixel space after denorm)."""

    default_prefix = 'fde'

    def process(self, data_batch: dict, data_samples: tuple):
        predictions, _, ground_truth, _, _ = data_samples

        final_pred = predictions[:, -1, :]
        final_gt = ground_truth[:, -1, :]

        width = 1920.0
        height = 1080.0
        final_pred = final_pred * torch.tensor([width, height], device=final_pred.device)
        final_gt = final_gt * torch.tensor([width, height], device=final_gt.device)

        fde_per_sample = torch.linalg.norm(final_pred - final_gt, dim=-1)

        self.results.append({
            'batch_size': len(ground_truth),
            'fde_sum': fde_per_sample.sum().cpu()
        })

    def compute_metrics(self, results: list) -> dict:
        total_fde = sum(item['fde_sum'] for item in results)
        total_samples = sum(item['batch_size'] for item in results)

        if total_samples == 0:
            return {self.prefix: 0.0}

        final_fde = total_fde / total_samples
        return {self.prefix: final_fde.item()}
