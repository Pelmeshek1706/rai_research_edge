# https://github.com/d-li14/mobilenetv3.pytorch

import torch.nn as nn
import brevitas.nn as qnn

from brevitas.quant import Int32Bias
from brevitas.quant import Int8Bias

from brevitas_examples.imagenet_classification.models.common import CommonIntWeightPerChannelQuant
from brevitas_examples.imagenet_classification.models.common import CommonIntWeightPerTensorQuant
from brevitas_examples.imagenet_classification.models.common import CommonUintActQuant

import math


__all__ = ['mobilenetv3_large', 'mobilenetv3_small']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
    
class QuantHsigmoid(nn.Module):
    def __init__(self, bit_width=8):
        super().__init__()
        # Quantize input and output
        self.quant_in = qnn.QuantIdentity(bit_width=bit_width, 
                                          act_quant=CommonUintActQuant, 
                                          return_quant_tensor=True)
        self.quant_out = qnn.QuantIdentity(bit_width=bit_width, 
                                           act_quant=CommonUintActQuant, 
                                           return_quant_tensor=True)
        # Quantized ReLU6 equivalent
        self.relu6 = qnn.QuantReLU(bit_width=bit_width, 
                                   max_val=6.0, 
                                   return_quant_tensor=True)

    def forward(self, x):
        x = self.quant_in(x)
        x = self.relu6(x + 3.0) / 6.0
        x = self.quant_out(x)
        return x

class QuantHswish(nn.Module):
    def __init__(self, bit_width=8):
        super().__init__()
        self.hsigmoid = QuantHsigmoid(bit_width)
        self.quant_out = qnn.QuantIdentity(bit_width=bit_width, 
                                           act_quant=CommonUintActQuant, 
                                           return_quant_tensor=True)

    def forward(self, x):
        x_sig = self.hsigmoid(x)
        x = x * x_sig
        x = self.quant_out(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4, weight_quant=CommonIntWeightPerTensorQuant, weight_bit_width=8, act_bit_width=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                qnn.QuantIdentity(
                    bit_width=act_bit_width,
                    act_quant=CommonUintActQuant,
                    return_quant_tensor=True
                ),
                qnn.QuantLinear(channel, _make_divisible(channel // reduction, 8),
                                bias=True,
                                bias_quant=Int8Bias,
                                weight_quant=weight_quant,
                                weight_bit_width=weight_bit_width,
                                return_quant_tensor=True),
                qnn.QuantReLU(inplace=True,
                              bit_width=act_bit_width,
                              act_quant=CommonUintActQuant,
                              return_quant_tensor=True),
                qnn.QuantLinear(_make_divisible(channel // reduction, 8), channel,
                            bias=True,
                            bias_quant=Int8Bias,
                            weight_quant=weight_quant,
                            weight_bit_width=weight_bit_width,
                            return_quant_tensor=True),
                QuantHswish(bit_width=act_bit_width)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride, weight_quant, weight_bit_width, act_bit_width):
    return nn.Sequential(
        qnn.QuantConv2d(inp, oup, 3, stride, 1, bias=False,
                        weight_quant=weight_quant,
                        weight_bit_width=weight_bit_width),
        nn.BatchNorm2d(oup),
        QuantHswish(bit_width=act_bit_width)
    )


def conv_1x1_bn(inp, oup, weight_quant, weight_bit_width, act_bit_width):
    return nn.Sequential(
        qnn.QuantConv2d(inp, oup, 1, 1, 0, bias=False, 
                        weight_quant=weight_quant,
                        weight_bit_width=weight_bit_width),
        nn.BatchNorm2d(oup),
        QuantHswish(bit_width=act_bit_width)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs, weight_quant, weight_bit_width, act_bit_width):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                qnn.QuantConv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False,
                                weight_quant=weight_quant,
                                weight_bit_width=weight_bit_width),
                nn.BatchNorm2d(hidden_dim),
                QuantHswish(bit_width=act_bit_width) if use_hs else qnn.QuantReLU(inplace=True,
                                                       bit_width=act_bit_width,
                                                       act_quant=CommonUintActQuant,
                                                       return_quant_tensor=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim, weight_bit_width=weight_bit_width) if use_se else qnn.QuantIdentity(return_quant_tensor=True),
                # pw-linear
                qnn.QuantConv2d(hidden_dim, oup, 1, 1, 0, bias=False,
                                weight_quant=weight_quant,
                                weight_bit_width=weight_bit_width),
                nn.BatchNorm2d(oup)
            )
        else:
            self.conv = nn.Sequential(
                # pw
                qnn.QuantConv2d(inp, hidden_dim, 1, 1, 0, bias=False,
                                weight_quant=weight_quant,
                                weight_bit_width=weight_bit_width),
                nn.BatchNorm2d(hidden_dim),
                QuantHswish(bit_width=act_bit_width) if use_hs else qnn.QuantReLU(inplace=True,
                                                       bit_width=act_bit_width,
                                                       act_quant=CommonUintActQuant,
                                                       return_quant_tensor=True),
                # dw
                qnn.QuantConv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False,
                               weight_quant=weight_quant,
                               weight_bit_width=weight_bit_width),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim, weight_bit_width=weight_bit_width) if use_se else qnn.QuantIdentity(return_quant_tensor=True),
                QuantHswish(bit_width=act_bit_width) if use_hs else qnn.QuantReLU(inplace=True,
                                                       bit_width=act_bit_width,
                                                       act_quant=CommonUintActQuant,
                                                       return_quant_tensor=True),
                # pw-linear
                qnn.QuantConv2d(hidden_dim, oup, 1, 1, 0, bias=False,
                          weight_quant=weight_quant,
                          weight_bit_width=weight_bit_width),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, cfgs, mode,
                 weight_bit_width,
                 weight_quant,
                 act_bit_width,
                 num_classes=10, width_mult=1.):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        assert mode in ['large', 'small']

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2, weight_quant, weight_bit_width, act_bit_width)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs, weight_quant, weight_bit_width, act_bit_width))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        self.conv = conv_1x1_bn(input_channel, exp_size, weight_quant, weight_bit_width, act_bit_width)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        output_channel = {'large': 1280, 'small': 1024}
        output_channel = _make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[mode]
        self.classifier = nn.Sequential(
            nn.Linear(exp_size, output_channel),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, qnn.QuantConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear) or isinstance(m, qnn.QuantLinear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv3_large(**kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,   1,  16, 0, 0, 1],
        [3,   4,  24, 0, 0, 2],
        [3,   3,  24, 0, 0, 1],
        [5,   3,  40, 1, 0, 2],
        [5,   3,  40, 1, 0, 1],
        [5,   3,  40, 1, 0, 1],
        [3,   6,  80, 0, 1, 2],
        [3, 2.5,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [5,   6, 160, 1, 1, 2],
        [5,   6, 160, 1, 1, 1],
        [5,   6, 160, 1, 1, 1]
    ]
    return MobileNetV3(cfgs, mode='large', **kwargs)


def mobilenetv3_small_quantized(weight_bit_width=8, 
                                weight_quant=CommonIntWeightPerChannelQuant,
                                act_bit_width=8,
                                **kwargs):
    """
    Constructs a MobileNetV3-Small model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,    1,  16, 1, 0, 2],
        [3,  4.5,  24, 0, 0, 2],
        [3, 3.67,  24, 0, 0, 1],
        [5,    4,  40, 1, 1, 2],
        [5,    6,  40, 1, 1, 1],
        [5,    6,  40, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    6,  96, 1, 1, 2],
        [5,    6,  96, 1, 1, 1],
        [5,    6,  96, 1, 1, 1],
    ]
    return MobileNetV3(cfgs, 'small',
                       weight_bit_width,
                       weight_quant, 
                       act_bit_width,
                        **kwargs)


# Let's use CommonIntWeightPerChannelQuant in CNN layers
# and CommonIntWeightPerTensorQuant in Linear layers