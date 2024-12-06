import torch
import torch.nn as nn

STYLE_LOSS_WEIGHT = 1e6
PERCEPTION_LOSS_WEIGHT = 1e0

def gram(x):
    b, c, h, w = x.shape
    tmp = x.view((b, c, h*w))
    tmp1 = tmp.transpose(1, 2)
    tmp2 = torch.matmul(tmp, tmp1)
    tmp2 = tmp2 / c / h / w
    return tmp2

def get_loss(y_hat_features, style_features, x_features):
    mse_loss = nn.MSELoss()
    style_loss = 0
    for i in range(4):
        G1 = gram(y_hat_features[i])
        b = y_hat_features[i].shape[0]
        style_feature = style_features[i].repeat(b, 1, 1, 1)
        G2 = gram(style_feature)
        style_loss += mse_loss(G1, G2)

    perception_loss = 0
    perception_loss = mse_loss(y_hat_features[2], x_features[2])

    return STYLE_LOSS_WEIGHT * style_loss + PERCEPTION_LOSS_WEIGHT * perception_loss, style_loss, perception_loss





