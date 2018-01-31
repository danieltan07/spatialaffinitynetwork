import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torch.autograd import Variable

# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - np.abs(og[0] - center) / factor) * \
           (1 - np.abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class FCN32s(nn.Module):
    def __init__(self, n_class=21):
        super(FCN32s, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.upscore = nn.ConvTranspose2d(n_class, n_class, 64, stride=32,
                                          bias=False)

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)

        h = self.upscore(h)
        h = h[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()

        return h

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def copy_params_from_vgg16(self, vgg16):
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
            self.pool5,
        ]
        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
        for i, name in zip([0, 3], ['fc6', 'fc7']):
            l1 = vgg16.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())



class FCN16s(nn.Module):

    # pretrained_model = \
    #     osp.expanduser('~/data/models/pytorch/fcn16s_from_caffe.pth')

    # @classmethod
    # def download(cls):
    #     return fcn.data.cached_download(
    #         url='http://drive.google.com/uc?id=0B9P1L--7Wd2vVGE3TkRMbWlNRms',
    #         path=cls.pretrained_model,
    #         md5='991ea45d30d632a01e5ec48002cac617',
    #     )

    def __init__(self, n_class=21):
        super(FCN16s, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)

        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.upscore16 = nn.ConvTranspose2d(
            n_class, n_class, 32, stride=16, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4)
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c

        h = self.upscore16(h)
        h = h[:, :, 27:27 + x.size()[2], 27:27 + x.size()[3]].contiguous()

        return h

    def copy_params_from_fcn32s(self, fcn32s):
        for name, l1 in fcn32s.named_children():
            try:
                l2 = getattr(self, name)
                l2.weight  # skip ReLU / Dropout
            except Exception:
                continue
            assert l1.weight.size() == l2.weight.size()
            assert l1.bias.size() == l2.bias.size()
            l2.weight.data.copy_(l1.weight.data)
            l2.bias.data.copy_(l1.bias.data)


class FCN8s(nn.Module):

    # pretrained_model = \
    #     osp.expanduser('~/data/models/pytorch/fcn8s_from_caffe.pth')

    # @classmethod
    # def download(cls):
    #     return fcn.data.cached_download(
    #         url='http://drive.google.com/uc?id=0B9P1L--7Wd2vT0FtdThWREhjNkU',
    #         path=cls.pretrained_model,
    #         md5='dbd9bbb3829a3184913bccc74373afbb',
    #     )

    def __init__(self, n_class=21):
        super(FCN8s, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.score_pool3 = nn.Conv2d(256, n_class, 1)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)

        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h  # 1/8

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4)
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8

        h = self.score_pool3(pool3)
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        return h

    def copy_params_from_fcn16s(self, fcn16s):
        for name, l1 in fcn16s.named_children():
            try:
                l2 = getattr(self, name)
                l2.weight  # skip ReLU / Dropout
            except Exception:
                continue
            assert l1.weight.size() == l2.weight.size()
            l2.weight.data.copy_(l1.weight.data)
            if l1.bias is not None:
                assert l1.bias.size() == l2.bias.size()
                l2.bias.data.copy_(l1.bias.data)


class FCN8sAtOnce(FCN8s):

    # pretrained_model = \
    #     osp.expanduser('~/data/models/pytorch/fcn8s-atonce_from_caffe.pth')

    # @classmethod
    # def download(cls):
    #     return fcn.data.cached_download(
    #         url='http://drive.google.com/uc?id=0B9P1L--7Wd2vblE1VUIxV1o2d2M',
    #         path=cls.pretrained_model,
    #         md5='bfed4437e941fef58932891217fe6464',
    #     )

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h  # 1/8

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4 * 0.01)  # XXX: scaling to train at once
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8

        h = self.score_pool3(pool3 * 0.0001)  # XXX: scaling to train at once
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        return h

    def copy_params_from_vgg16(self, vgg16):
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
            self.pool5,
        ]
        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data.copy_(l1.weight.data)
                l2.bias.data.copy_(l1.bias.data)
        for i, name in zip([0, 3], ['fc6', 'fc7']):
            l1 = vgg16.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data.copy_(l1.weight.data.view(l2.weight.size()))
            l2.bias.data.copy_(l1.bias.data.view(l2.bias.size()))

class VGG16Modified(nn.Module):

    def __init__(self, n_classes=21):
        super(VGG16Modified, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2 128

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4 64

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8 32

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16 16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32 8


        self.conv6s_re = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
                            nn.ReLU(inplace=True),
                            nn.Upsample(scale_factor=2, mode='bilinear'))
        self.conv6_3 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
                            nn.ReLU(inplace=True))
        self.conv6_2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
                            nn.ReLU(inplace=True))
        self.conv6_1 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
                            nn.ReLU(inplace=True),
                            nn.Upsample(scale_factor=2, mode='bilinear'))

        self.conv7_3 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
                            nn.ReLU(inplace=True))
        self.conv7_2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
                            nn.ReLU(inplace=True))
        self.conv7_1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
                            nn.ReLU(inplace=True),
                            nn.Upsample(scale_factor=2, mode='bilinear'))

        self.conv8_3 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                            nn.ReLU(inplace=True))
        self.conv8_2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                            nn.ReLU(inplace=True))
        self.conv8_1 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True),
                            nn.ReLU(inplace=True),
                            nn.Upsample(scale_factor=2, mode='bilinear'))

        self.conv9 = nn.Sequential(nn.Conv2d(128, 32*3*4, kernel_size=3, stride=1, padding=1, bias=True),
                            nn.Tanh())

        self.conv10 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
                            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(nn.Conv2d(64, n_classes, kernel_size=3, stride=1, padding=1, bias=True),
                            nn.Upsample(scale_factor=2, mode='bilinear'))



        self.coarse_conv_in = nn.Sequential(nn.Conv2d(n_classes, 32, kernel_size=3, stride=1, padding=1, bias=True),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
                            nn.ReLU(inplace=True),
                            nn.AvgPool2d(kernel_size=2, stride=2))


    def to_tridiagonal_multidim(self, w):
        N,W,C,D = w.size()
        tmp_w = w.unsqueeze(2).expand([N,W,W,C,D])

        eye_a = Variable(torch.diag(torch.ones(W-1).cuda(),diagonal=-1))
        eye_b = Variable(torch.diag(torch.ones(W).cuda(),diagonal=0))
        eye_c = Variable(torch.diag(torch.ones(W-1).cuda(),diagonal=1))

        
        tmp_eye_a = eye_a.unsqueeze(-1).unsqueeze(0).expand([N,W,W,C])
        a = tmp_w[:,:,:,:,0] * tmp_eye_a
        tmp_eye_b = eye_b.unsqueeze(-1).unsqueeze(0).expand([N,W,W,C])
        b = tmp_w[:,:,:,:,1] * tmp_eye_b
        tmp_eye_c = eye_c.unsqueeze(-1).unsqueeze(0).expand([N,W,W,C])
        c = tmp_w[:,:,:,:,2] * tmp_eye_c

        return a+b+c
    def forward(self, x, coarse_segmentation):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        conv3_1 = self.relu3_1(self.conv3_1(h))
        conv3_2 = self.relu3_2(self.conv3_2(conv3_1))
        conv3_3= self.relu3_3(self.conv3_3(conv3_2))
        h = self.pool3(conv3_3)
        pool3 = h  # 1/8

        conv4_1 = self.relu4_1(self.conv4_1(h))
        conv4_2 = self.relu4_2(self.conv4_2(conv4_1))
        conv4_3 = self.relu4_3(self.conv4_3(conv4_2))
        h = self.pool4(conv4_3)
        pool4 = h  # 1/16

        conv5_1 = self.relu5_1(self.conv5_1(h))
        conv5_2 = self.relu5_2(self.conv5_2(conv5_1))
        conv5_3 = self.relu5_3(self.conv5_3(conv5_2))
        h = self.pool5(conv5_3)



        conv6_re = self.conv6s_re(h)



        skip_1 = conv5_3 + conv6_re
        conv6_3 = self.conv6_3(skip_1)        
        skip_2 = conv5_2 + conv6_3
        conv6_2 = self.conv6_2(skip_2)        
        skip_3 = conv5_1 + conv6_2
        conv6_1 = self.conv6_1(skip_3)

        skip_4 = conv4_3 + conv6_1
        conv7_3 = self.conv7_3(skip_4)        
        skip_5 = conv4_2 + conv7_3
        conv7_2 = self.conv7_2(skip_5)        
        skip_6 = conv4_1 + conv7_2
        conv7_1 = self.conv7_1(skip_6)

        skip_7 = conv3_3 + conv7_1
        conv8_3 = self.conv8_3(skip_7)        
        skip_8 = conv3_2 + conv8_3
        conv8_2 = self.conv8_2(skip_8)        
        skip_9 = conv3_1 + conv8_2
        conv8_1 = self.conv8_1(skip_9)

        conv9 = self.conv9(conv8_1)

        N,C,H,W = conv9.size()
        four_directions = C // 4
        conv9_reshaped_W = conv9.permute(0,2,3,1)
        # conv9_reshaped_H = conv9.permute(0,3,2,1)

        conv_x1_flat = conv9_reshaped_W[:,:,:,0:four_directions].contiguous()
        conv_y1_flat = conv9_reshaped_W[:,:,:,four_directions:2*four_directions].contiguous()
        conv_x2_flat = conv9_reshaped_W[:,:,:,2*four_directions:3*four_directions].contiguous()
        conv_y2_flat = conv9_reshaped_W[:,:,:,3*four_directions:4*four_directions].contiguous()

        w_x1 = conv_x1_flat.view(N,H,W,four_directions//3,3) # N, H, W, 32, 3
        w_y1 = conv_y1_flat.view(N,H,W,four_directions//3,3) # N, H, W, 32, 3
        w_x2 = conv_x2_flat.view(N,H,W,four_directions//3,3) # N, H, W, 32, 3
        w_y2 = conv_y2_flat.view(N,H,W,four_directions//3,3) # N, H, W, 32, 3

        rnn_h1 = Variable(torch.zeros((N, H, W, four_directions//3)).cuda())
        rnn_h2 = Variable(torch.zeros((N, H, W, four_directions//3)).cuda())
        rnn_h3 = Variable(torch.zeros((N, H, W, four_directions//3)).cuda())
        rnn_h4 = Variable(torch.zeros((N, H, W, four_directions//3)).cuda())

        x_t = self.coarse_conv_in(coarse_segmentation).permute(0,2,3,1)

        # horizontal
        for i in range(W):
            #left to right
            tmp_w = w_x1[:,:,i,:,:]  # N, H, 1, 32, 3 
            tmp_w = self.to_tridiagonal_multidim(tmp_w) # N, H, W, 32 
            # tmp_x = x_t[:,:,i,:].unsqueeze(1)
            # tmp_x = tmp_x.expand([batch, W, H, 32])

            w_h_prev_1 = torch.sum(tmp_w * rnn_h1[:,:,i-1,:].clone().unsqueeze(1).expand([N, W, H, 32]),dim=2)
            w_x_curr_1 = (1 - torch.sum(tmp_w, dim=2)) * x_t[:,:,i,:]
            rnn_h1[:,:,i,:] = w_x_curr_1 + w_h_prev_1


            #right to left
            # tmp_w = w_x1[:,:,i,:,:]  # N, H, 1, 32, 3 
            # tmp_w = to_tridiagonal_multidim(tmp_w)
            w_h_prev_2 = torch.sum(tmp_w * rnn_h2[:,:,i-1,:].clone().unsqueeze(1).expand([N, W, H, 32]),dim=2)
            w_x_curr_2 = (1 - torch.sum(tmp_w, dim=2)) * x_t[:,:,W - i-1,:]
            rnn_h2[:,:,i,:] = w_x_curr_2 + w_h_prev_2   

        w_y1_T = w_y1.transpose(1,2)
        x_t_T = x_t.transpose(1,2)

        for i in range(H):
            # up to down
            tmp_w = w_y1_T[:,:,i,:,:]  # N, W, 1, 32, 3 
            tmp_w = self.to_tridiagonal_multidim(tmp_w) # N, W, H, 32 

            w_h_prev_3 = torch.sum(tmp_w * rnn_h3[:,:,i-1,:].clone().unsqueeze(1).expand([N, H, W, 32]),dim=2)
            w_x_curr_3 = (1 - torch.sum(tmp_w, dim=2)) * x_t_T[:,:,i,:]
            rnn_h3[:,:,i,:] = w_x_curr_3 + w_h_prev_3

            # down to up
            w_h_prev_4 = torch.sum(tmp_w * rnn_h4[:,:,i-1,:].clone().unsqueeze(1).expand([N, H, W, 32]),dim=2)
            w_x_curr_4 = (1 - torch.sum(tmp_w, dim=2)) * x_t[:,:,H-i-1,:]
            rnn_h4[:,:,i,:] = w_x_curr_4 + w_h_prev_4   

        rnn_h3 = rnn_h3.transpose(1,2)
        rnn_h4 = rnn_h4.transpose(1,2)



        rnn_h5 = Variable(torch.zeros((N, H, W, four_directions//3)).cuda())
        rnn_h6 = Variable(torch.zeros((N, H, W, four_directions//3)).cuda())
        rnn_h7 = Variable(torch.zeros((N, H, W, four_directions//3)).cuda())
        rnn_h8 = Variable(torch.zeros((N, H, W, four_directions//3)).cuda())

        # horizontal
        for i in range(W):
            #left to right
            tmp_w = w_x2[:,:,i,:,:]  # N, H, 1, 32, 3 
            tmp_w = self.to_tridiagonal_multidim(tmp_w) # N, H, W, 32 
            # tmp_x = x_t[:,:,i,:].unsqueeze(1)
            # tmp_x = tmp_x.expand([batch, W, H, 32])

            w_h_prev_5 = torch.sum(tmp_w * rnn_h5[:,:,i-1,:].clone().unsqueeze(1).expand([N, W, H, 32]),dim=2)
            w_x_curr_5 = (1 - torch.sum(tmp_w, dim=2)) * rnn_h1[:,:,i,:]
            rnn_h5[:,:,i,:] = w_x_curr_5 + w_h_prev_5


            #right to left
            # tmp_w = w_x1[:,:,i,:,:]  # N, H, 1, 32, 3 
            # tmp_w = to_tridiagonal_multidim(tmp_w)
            w_h_prev_6 = torch.sum(tmp_w * rnn_h6[:,:,i-1,:].clone().unsqueeze(1).expand([N, W, H, 32]),dim=2)
            w_x_curr_6 = (1 - torch.sum(tmp_w, dim=2)) * rnn_h2[:,:,W - i-1,:]
            rnn_h6[:,:,i,:] = w_x_curr_6 + w_h_prev_6   

        w_y2_T = w_y2.transpose(1,2)
        rnn_h3_T = rnn_h3.transpose(1,2)
        rnn_h4_T = rnn_h4.transpose(1,2)
        for i in range(H):
            # up to down
            tmp_w = w_y2_T[:,:,i,:,:]  # N, W, 1, 32, 3 
            tmp_w = self.to_tridiagonal_multidim(tmp_w) # N, W, H, 32 

            w_h_prev_7 = torch.sum(tmp_w * rnn_h7[:,:,i-1,:].clone().unsqueeze(1).expand([N, H, W, 32]),dim=2)
            w_x_curr_7 = (1 - torch.sum(tmp_w, dim=2)) * rnn_h3_T[:,:,i,:]
            rnn_h7[:,:,i,:] = w_x_curr_7 + w_h_prev_7

            # down to up
            w_h_prev_8 = torch.sum(tmp_w * rnn_h8[:,:,i-1,:].clone().unsqueeze(1).expand([N, H, W, 32]),dim=2)
            w_x_curr_8 = (1 - torch.sum(tmp_w, dim=2)) * rnn_h4_T[:,:,H-i-1,:]
            rnn_h8[:,:,i,:] = w_x_curr_8 + w_h_prev_8   

        rnn_h3 = rnn_h3.transpose(1,2)
        rnn_h4 = rnn_h4.transpose(1,2)

        concat6 = torch.cat([rnn_h5.unsqueeze(4),rnn_h6.unsqueeze(4),rnn_h7.unsqueeze(4),rnn_h8.unsqueeze(4)],dim=4)
        elt_max = torch.max(concat6, dim=4)[0]
        elt_max_reordered = elt_max.permute(0,3,1,2)
        conv10 = self.conv10(elt_max_reordered)
        conv11 = self.conv11(conv10)
        return conv11

    def copy_params_from_vgg16(self, vgg_model_file):
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
            self.pool5,
        ]


        vgg16 = torchvision.models.vgg16(pretrained=False)
        state_dict = torch.load(vgg_model_file)
        vgg16.load_state_dict(state_dict)


        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data.copy_(l1.weight.data)
                l2.bias.data.copy_(l1.bias.data)

