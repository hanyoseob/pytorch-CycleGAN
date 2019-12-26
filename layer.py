import torch
import torch.nn as nn
import torch.nn.functional as F


class Pooling2d(nn.Module):
    def __init__(self, nch=[], pool=2, type='avg'):
        super().__init__()

        if type == 'avg':
            self.pooling = nn.AvgPool2d(pool)
        elif type == 'max':
            self.pooling = nn.MaxPool2d(pool)
        elif type == 'conv':
            self.pooling = nn.Conv2d(nch, nch, kernel_size=pool, stride=pool)

    def forward(self, x):
        return self.pooling(x)


class UnPooling2d(nn.Module):
    def __init__(self, nch=[], pool=2, type='nearest'):
        super().__init__()

        if type == 'nearest':
            self.unpooling = nn.Upsample(scale_factor=pool, mode='nearest', align_corners=True)
        elif type == 'bilinear':
            self.unpooling = nn.Upsample(scale_factor=pool, mode='bilinear', align_corners=True)
        elif type == 'conv':
            self.unpooling = nn.ConvTranspose2d(nch, nch, kernel_size=pool, stride=pool)

    def forward(self, x):
        return self.unpooling(x)


class Concat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        diffy = x2.size()[2] - x1.size()[2]
        diffx = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffx // 2, diffx - diffx // 2,
                        diffy // 2, diffy - diffy // 2])

        return torch.cat([x2, x1], dim=1)


class CNR1d(nn.Module):
    def __init__(self, nch_in, nch_out, bnorm=True, brelu=True):
        super().__init__()

        layers = nn.Linear(nch_in, nch_out)
        if bnorm:
            layers.append(nn.InstanceNorm1d(nch_out))
        if brelu:
            layers.append(nn.LeakyReLU(brelu))

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)


class CNR2d(nn.Module):
    def __init__(self, nch_in, nch_out, kerner_size=4, stride=1, padding=1, norm='bnorm', relu=0.0, drop=0.0):
        super().__init__()

        if norm == 'bnorm':
            bias = False
        else:
            bias = True

        layers = [nn.Conv2d(nch_in, nch_out, kernel_size=kerner_size, stride=stride, padding=padding, bias=bias)]

        if norm == 'bnorm':
            layers.append(nn.BatchNorm2d(nch_out))
        elif norm == 'inorm':
            layers.append(nn.InstanceNorm2d(nch_out))

        if relu != 0.0:
            layers.append(nn.LeakyReLU(relu, True))
        elif relu == 0.0:
            layers.append(nn.ReLU(True))

        if drop:
            layers.append(nn.Dropout2d(drop))

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)


class DECNR2d(nn.Module):
    def __init__(self, nch_in, nch_out, kerner_size=4, stride=1, padding=1, norm='bnorm', relu=0.0, drop=0.0):
        super().__init__()

        if norm == 'bnorm':
            bias = False
        else:
            bias = True

        layers = [nn.ConvTranspose2d(nch_in, nch_out, kernel_size=kerner_size, stride=stride, padding=padding, bias=bias)]

        if norm == 'bnorm':
            layers.append(nn.BatchNorm2d(nch_out))
        elif norm == 'inorm':
            layers.append(nn.InstanceNorm2d(nch_out))

        if relu:
            layers.append(nn.LeakyReLU(relu, True))
        else:
            layers.append(nn.ReLU(True))

        if drop:
            layers.append(nn.Dropout2d(drop))

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)


class Conv2d(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=4, stride=1, padding=1):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.conv(x)


class Deconv2d(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=4, stride=1, padding=1):
        super(Deconv2d, self).__init__()
        self.deconv = nn.ConvTranspose2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.deconv(x)


class Linear(nn.Module):
    def __init__(self, nch_in, nch_out):
        super(Linear, self).__init__()
        self.linear = nn.Linear(nch_in, nch_out)

    def forward(self, x):
        return self.linear(x)


class TV1dLoss(nn.Module):
    def __init__(self):
        super(TV1dLoss, self).__init__()

    def forward(self, input):
        # loss = torch.mean(torch.abs(input[:, :, :, :-1] - input[:, :, :, 1:])) + \
        #        torch.mean(torch.abs(input[:, :, :-1, :] - input[:, :, 1:, :]))
        loss = torch.mean(torch.abs(input[:, :-1] - input[:, 1:]))

        return loss


class TV2dLoss(nn.Module):
    def __init__(self):
        super(TV2dLoss, self).__init__()

    def forward(self, input):
        loss = torch.mean(torch.abs(input[:, :, :, :-1] - input[:, :, :, 1:])) + \
               torch.mean(torch.abs(input[:, :, :-1, :] - input[:, :, 1:, :]))
        return loss


class SSIM2dLoss(nn.Module):
    def __init__(self):
        super(SSIM2dLoss, self).__init__()

    def forward(self, input, targer):
        loss = 0
        return loss

