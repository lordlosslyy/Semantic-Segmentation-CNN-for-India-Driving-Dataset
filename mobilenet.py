import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class reverse_block(nn.Module):
    def __init__(self, input, output, stride, dilation, ratio):
        super(reverse_block, self).__init__() 
        self.stride = stride

        hidden_dim = round(input * ratio)
        self.reverse = self.stride == 1 and input == output
        self.kernel_size = 3
        self.dilation = dilation

        if ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 0, dilation, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, output, 1, 1, 0, 1, 1, bias=False),
                nn.BatchNorm2d(output),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(input, hidden_dim, 1, 1, 0, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 0, dilation, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, output, 1, 1, 0, 1, bias=False),
                nn.BatchNorm2d(output),
            )

    def forward(self, x):
        x_pad = fixed_padding(x, self.kernel_size, self.dilation)
        if self.reverse:
            x = x + self.conv(x_pad)
        else:
            x = self.conv(x_pad)
        return x


class MobileNetV2(nn.Module):
    def __init__(self, output_stride=8,  pretrained=True):
        super(MobileNetV2, self).__init__()
        block = reverse_block
        input_channel = 32
        current_stride = 1
        rate = 1
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
    
        # building first layer
        input_channel = input_channel
        self.conv_block = nn.Sequential(nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                                        nn.BatchNorm2d(input_channel),
                                        nn.ReLU6(inplace=True) ) 
        self.features = [self.conv_block]
        current_stride *= 2
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            if current_stride == output_stride:
                stride = 1
                dilation = rate
                rate *= s
            else:
                stride = s
                dilation = 1
                current_stride *= s
            output_channel = c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, stride, dilation, t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, dilation, t))
                input_channel = output_channel
        self.features = nn.Sequential(*self.features)
        self._initialize_weights()

        if pretrained:
            self._load_pretrained_model()

        self.low_level_features = self.features[0:4]
        self.high_level_features = self.features[4:]

    def forward(self, x):
        low_level_feat = self.low_level_features(x)
        x = self.high_level_features(low_level_feat)
        return x, low_level_feat

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('http://jeff95.me/models/mobilenet_v2-6a65762b.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
