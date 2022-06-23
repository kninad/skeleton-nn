import torch
import torch.nn as nn
import torch.nn.functional as nnFunc


class StructuredLoss(nn.Module):
    def __init__(self) -> None:
        super(StructuredLoss, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=31, stride=1, padding=15)

    def forward(self, pred, mask):
        weit = 1 + 5 * torch.abs(self.pool(mask) - mask)
        wbce = nnFunc.binary_cross_entropy_with_logits(
            pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        wdiss = 1 - (2*inter + 0.5)/(union + 0.5)
        return (wbce + wiou + wdiss).mean()


class DiceLoss(nn.Module):
    def __init__(self) -> None:
        super(DiceLoss, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=31, stride=1, padding=15)

    def forward(self, pred, mask):
        weit = 1 + 5 * torch.abs(self.pool(mask) - mask)
        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wdiss = 1 - (2*inter + 0.5)/(union+0.5)
        return wdiss.mean()


# Reference: https://amaarora.github.io/2020/06/29/FocalLoss.html
class WeightedFocalLoss(nn.Module):
    # Weighted version of Focal Loss
    def __init__(self, alpha=[1, 0.25], gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nnFunc.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
