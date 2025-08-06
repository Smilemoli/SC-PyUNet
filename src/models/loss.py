import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()
        intersection = (pred * target).sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + self.smooth
        )
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        # 计算BCE损失
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

        # 计算pt
        pred_sigmoid = torch.sigmoid(pred)
        pt = torch.where(target == 1, pred_sigmoid, 1 - pred_sigmoid)

        # 计算focal权重
        alpha = torch.where(target == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha * (1 - pt).pow(self.gamma)

        # 使用detach()确保权重不需要梯度
        focal_loss = focal_weight.detach() * bce_loss

        return focal_loss.mean()


class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def compute_distance_map(self, mask):
        """使用CV2计算距离变换"""
        batch_size = mask.size(0)
        distance_maps = []

        # 转换为numpy处理
        masks_np = mask.cpu().numpy()

        for i in range(batch_size):
            # 提取单个mask
            curr_mask = masks_np[i, 0].astype(np.uint8)

            # 计算正负距离图
            pos_dist = cv2.distanceTransform(curr_mask, cv2.DIST_L2, 3)
            neg_dist = cv2.distanceTransform(1 - curr_mask, cv2.DIST_L2, 3)

            # 归一化
            pos_dist = pos_dist / (pos_dist.max() + 1e-8)
            neg_dist = neg_dist / (neg_dist.max() + 1e-8)

            # 转回tensor
            pos_dist = torch.from_numpy(pos_dist).float()
            neg_dist = torch.from_numpy(neg_dist).float()

            distance_maps.append((pos_dist, neg_dist))

        # 堆叠成batch
        pos_maps = torch.stack([d[0] for d in distance_maps]).unsqueeze(1)
        neg_maps = torch.stack([d[1] for d in distance_maps]).unsqueeze(1)

        return pos_maps.to(mask.device), neg_maps.to(mask.device)

    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        pos_dist, neg_dist = self.compute_distance_map(target)

        # 计算边界损失
        boundary_loss = torch.mean(
            pred_sigmoid * neg_dist + (1 - pred_sigmoid) * pos_dist
        )
        return boundary_loss


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal = FocalLoss()
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss()

    def forward(self, pred, target):
        focal_loss = self.focal(pred, target)
        dice_loss = self.dice(torch.sigmoid(pred), target)
        boundary_loss = self.boundary(pred, target)

        return 0.3 * focal_loss + 0.4 * dice_loss + 0.3 * boundary_loss
