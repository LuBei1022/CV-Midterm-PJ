import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: [Batch, C, H, W] 模型输出
        targets: [Batch, H, W] 真实标签 (0, 1, 2)
        """
        num_classes = logits.shape[1]
        # 1. 将 logits 转换为概率分布
        probs = F.softmax(logits, dim=1)
        
        # 2. 将 targets 转换为 one-hot 编码 [Batch, C, H, W]
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # 3. 计算每个通道(类别)的 Dice
        dims = (0, 2, 3) # 在 Batch, H, W 维度上求和
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)
        
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        
        # 返回 1 - Dice 作为损失，取所有类别的平均值
        return 1 - dice_score.mean()