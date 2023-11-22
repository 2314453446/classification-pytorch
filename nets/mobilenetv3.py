from torch import nn
# from torchvision.models.utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url  # 由于pytorch版本问题更改上一行代码

__all__ = ['MobileNetV3', 'mobilenetv3']

model_urls = {
    'mobilenetv2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV3(nn.Module):
    def __init__(self, width_mult=1.0, num_classes=1000):
        super(MobileNetV3, self).__init__()

        input_channel = 32
        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, 8)
        features = [ConvBNReLU(3, input_channel, stride=2)]

        # Stop before the first inverted residual block
        self.features = nn.Sequential(*features)

        # Add the final classification layer
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(input_channel, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

def mobilenetv3(pretrained=False, progress=True, num_classes=1000):
    model = MobileNetV3()
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenetv2'], model_dir='./model_data',
                                            progress=progress)
        model.load_state_dict(state_dict)

    return model

