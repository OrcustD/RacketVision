import torch
import torch.nn as nn

def weighted_mse_loss(prediction, target, weights):
    """
    计算加权的MSE Loss。
    
    Args:
        prediction (torch.Tensor): 模型的预测输出，shape (batch_size, pred_len, 2)。
        target (torch.Tensor): 真实的标签，shape (batch_size, pred_len, 2)。
        weights (torch.Tensor): 应用于每个时间步的权重，shape (pred_len,)。
    
    Returns:
        torch.Tensor: 一个标量，表示最终的加权损失。
    """
    # 1. 初始化MSELoss，使其返回每个元素的误差，而不是平均值
    loss_fn = nn.MSELoss(reduction='none')
    
    # 2. 计算每个点的平方误差
    # elementwise_loss 的 shape 会是 (batch_size, pred_len, 2)
    elementwise_loss = loss_fn(prediction, target)
    
    # 3. 准备权重张量以便广播
    # 将 weights 的 shape 从 (pred_len,) 变为 (1, pred_len, 1)
    # 这样它就可以和 (batch_size, pred_len, 2) 的 loss 张量进行广播乘法
    weights_reshaped = weights.view(1, -1, 1).to(prediction.device)
    
    # 4. 将误差与权重相乘
    # (batch_size, pred_len, 2) * (1, pred_len, 1) -> (batch_size, pred_len, 2)
    weighted_loss_elements = elementwise_loss * weights_reshaped
    
    # 5. 计算所有加权误差的平均值
    final_loss = weighted_loss_elements.mean()
    
    return final_loss

# --- 示例用法 ---
if __name__ == '__main__':
    # 定义超参数
    BATCH_SIZE = 4
    PRED_LEN_LONG = 20  # 对应 M=20 的 setting
    PRED_LEN_SHORT = 5   # 对应 M=5 的 setting

    # 假设的模型输出和真实标签
    dummy_prediction = torch.randn(BATCH_SIZE, PRED_LEN_LONG, 2)
    dummy_target = torch.randn(BATCH_SIZE, PRED_LEN_LONG, 2)

    # --- 1. 为 M=20 的长预测创建线性衰减权重 ---
    # 权重从1.0线性衰减到0.1
    weights_long = torch.linspace(1.0, 0.1, PRED_LEN_LONG)
    print(f"Weights for M=20 (first 5): {weights_long[:5]}")
    print(f"Weights for M=20 (last 5): {weights_long[-5:]}")
    
    # 计算加权Loss
    loss_long = weighted_mse_loss(dummy_prediction, dummy_target, weights_long)
    print(f"\nWeighted MSE Loss (M=20): {loss_long.item()}")

    # 对比一下不加权的Loss
    unweighted_loss = nn.MSELoss()(dummy_prediction, dummy_target)
    print(f"Unweighted MSE Loss (M=20): {unweighted_loss.item()}")
    
    print("-" * 30)
    
    # --- 2. 为 M=5 的短预测创建权重 ---
    weights_short = torch.linspace(1.0, 0.5, PRED_LEN_SHORT)
    print(f"Weights for M=5: {weights_short}")
    
    dummy_prediction_short = torch.randn(BATCH_SIZE, PRED_LEN_SHORT, 2)
    dummy_target_short = torch.randn(BATCH_SIZE, PRED_LEN_SHORT, 2)
    
    loss_short = weighted_mse_loss(dummy_prediction_short, dummy_target_short, weights_short)
    print(f"\nWeighted MSE Loss (M=5): {loss_short.item()}")