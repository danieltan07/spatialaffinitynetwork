import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LRNN(nn.Module):

    def __init__(self):
        super(LRNN, self).__init__()
        
        self.scale_1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.scale_2 = nn.MaxPool2d(2, stride=2) 
        self.scale_3 = nn.MaxPool2d(2, stride=2) 
        self.scale_4 = nn.MaxPool2d(2, stride=2)

        self.scale_2_resize = nn.Upsample(scale_factor=2, mode='bilinear')
        self.scale_3_resize = nn.Upsample(scale_factor=4, mode='bilinear')
        self.scale_4_resize = nn.Upsample(scale_factor=8, mode='bilinear')
        self.multi_conv = nn.Conv2d(12, 16, kernel_size=3, stride=1, padding=1, bias=True)



        self.conv2 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2, bias=True),
                                    nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(nn.MaxPool2d(2, stride=2),
                                    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True),
                                    nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(nn.MaxPool2d(2, stride=2),
                                    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
                                    nn.ReLU(inplace=True))        

        self.conv5 = nn.Sequential(nn.MaxPool2d(2, stride=2),
                                    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
                                    nn.ReLU(inplace=True))        

        self.conv6 = nn.Sequential(nn.MaxPool2d(2, stride=2),
                                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                    nn.ReLU(inplace=True))

        self.conv6s_re = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                    nn.ReLU(inplace=True),
                                    nn.Upsample(scale_factor=2, mode='bilinear'))

        # concat conv6s_re with conv4
        self.conv7_re = nn.Sequential(nn.Conv2d(96, 32, kernel_size=3, stride=1, padding=1, bias=True),
                                    nn.ReLU(inplace=True),
                                    nn.Upsample(scale_factor=2, mode='bilinear'))
        # concat conv7_re with conv 3
        self.conv8_re = nn.Sequential(nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1, bias=True),
                                    nn.ReLU(inplace=True),
                                    nn.Upsample(scale_factor=2, mode='bilinear'))        

        # concat conv8_re with conv 2
        self.conv9 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                    nn.Tanh())  


        self.conv10 = nn.Sequential(nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                    nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=True),
                                    nn.ReLU(inplace=True))

    def sample_multiscale(self,x):
        scale_1 = self.scale_1(x)
        scale_2 = self.scale_2(scale_1)
        scale_3 = self.scale_3(scale_2)
        scale_4 = self.scale_4(scale_3)

        scale_2 = self.scale_2_resize(scale_2)
        scale_3 = self.scale_3_resize(scale_3)
        scale_4 = self.scale_4_resize(scale_4)
        multi = torch.cat([scale_1, scale_2, scale_3, scale_4], dim=1)

        return self.multi_conv(multi)

    def flip(self,x, dim):
        xsize = x.size()
        dim = x.dim() + dim if dim < 0 else dim
        x = x.view(-1, *xsize[dim:])
        x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                          -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
        return x.view(xsize)

    def forward(self, x):
        multiscale_input = self.sample_multiscale(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)       
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        conv6s_re = self.conv6s_re(conv6)

        concat3 = torch.cat([conv6s_re, conv4],dim=1)
        conv7_re = self.conv7_re(concat3)

        concat4 = torch.cat([conv7_re, conv3],dim=1)
        conv8_re = self.conv8_re(concat4)

        concat5 = torch.cat([conv8_re, conv2], dim=1)
        conv9 = self.conv9(concat5)
        
        conv4_bn_x1 = conv9[:,0:16,:,:]
        conv4_bn_y1 = conv9[:,16:32,:,:]
        conv4_bn_x2 = conv9[:,32:48,:,:]
        conv4_bn_y2 = conv9[:,48:64,:,:]

        N, C, H, W = x.size()


        rnn_h1 = ((1- conv4_bn_x1[:,:,:,0])*multiscale_input[:,:,:,0]).unsqueeze(3)
        rnn_h2 = ((1- conv4_bn_x1[:,:,:,0])*multiscale_input[:,:,:,W-1]).unsqueeze(3)
        rnn_h3 = ((1- conv4_bn_y1[:,:,0,:])*multiscale_input[:,:,0,:]).unsqueeze(2)
        rnn_h4 = ((1- conv4_bn_y1[:,:,0,:])*multiscale_input[:,:,H-1,:]).unsqueeze(2)

        for i in range(1,W):
            rnn_h1_t = conv4_bn_x1[:,:,:,i]*rnn_h1[:,:,:,i-1] + (1 - conv4_bn_x1[:,:,:,i])*multiscale_input[:,:,:,i]
            rnn_h2_t = conv4_bn_x1[:,:,:,i]*rnn_h2[:,:,:,i-1] + (1 - conv4_bn_x1[:,:,:,i])*multiscale_input[:,:,:,W-i-1]

            rnn_h1 = torch.cat([rnn_h1, rnn_h1_t.unsqueeze(3)], dim=3)
            rnn_h2 = torch.cat([rnn_h2, rnn_h2_t.unsqueeze(3)], dim=3)

        for i in range(1,H):
            rnn_h3_t = conv4_bn_x1[:,:,i,:]*rnn_h3[:,:,i-1,:] + (1 - conv4_bn_x1[:,:,i,:])*multiscale_input[:,:,i,:]
            rnn_h4_t = conv4_bn_x1[:,:,i,:]*rnn_h4[:,:,i-1,:] + (1 - conv4_bn_x1[:,:,i,:])*multiscale_input[:,:,H-i-1,:]
            
            rnn_h3 = torch.cat([rnn_h3, rnn_h3_t.unsqueeze(2)], dim=2)
            rnn_h4 = torch.cat([rnn_h4, rnn_h4_t.unsqueeze(2)], dim=2)
     
        rnn_h5 = ((1- conv4_bn_x2[:,:,:,0])*rnn_h1[:,:,:,0]).unsqueeze(3)
        rnn_h6 = ((1- conv4_bn_x2[:,:,:,0])*rnn_h2[:,:,:,W-1]).unsqueeze(3)
        rnn_h7 = ((1- conv4_bn_y2[:,:,0,:])*rnn_h3[:,:,0,:]).unsqueeze(2)
        rnn_h8 = ((1- conv4_bn_y2[:,:,0,:])*rnn_h4[:,:,H-1,:]).unsqueeze(2)

        for i in range(1,W):
            rnn_h5_t = conv4_bn_x2[:,:,:,i]*rnn_h5[:,:,:,i-1] + (1 - conv4_bn_x2[:,:,:,i])*rnn_h1[:,:,:,i]
            rnn_h6_t = conv4_bn_x2[:,:,:,i]*rnn_h6[:,:,:,i-1] + (1 - conv4_bn_x2[:,:,:,i])*rnn_h2[:,:,:,W-i-1]
            rnn_h5 = torch.cat([rnn_h5, rnn_h5_t.unsqueeze(3)], dim=3)
            rnn_h6 = torch.cat([rnn_h6, rnn_h6_t.unsqueeze(3)], dim=3)
        for i in range(1,H):
            rnn_h7_t = conv4_bn_y2[:,:,i,:]*rnn_h7[:,:,i-1,:] + (1 - conv4_bn_y2[:,:,i,:])*rnn_h3[:,:,i,:]
            rnn_h8_t = conv4_bn_y2[:,:,i,:]*rnn_h8[:,:,i-1,:] + (1 - conv4_bn_y2[:,:,i,:])*rnn_h4[:,:,H-i-1,:]

            rnn_h7 = torch.cat([rnn_h7, rnn_h7_t.unsqueeze(2)], dim=2)
            rnn_h8 = torch.cat([rnn_h8, rnn_h8_t.unsqueeze(2)], dim=2)
        concat6 = torch.cat([rnn_h5.unsqueeze(4),rnn_h6.unsqueeze(4),rnn_h7.unsqueeze(4),rnn_h8.unsqueeze(4)],dim=4)
        elt_max = torch.max(concat6, dim=4)[0]
        conv10 = self.conv10(elt_max)
        conv11 = self.conv11(conv10)

        return conv11

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

class SimpleRNN(nn.Module):
    def __init__(self, hidden_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size

        self.inp = nn.Linear(1, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, 2, dropout=0.05)
        self.out = nn.Linear(hidden_size, 1)

    def step(self, input, hidden=None):
        input = self.inp(input.view(1, -1)).unsqueeze(1)
        output, hidden = self.rnn(input, hidden)
        output = self.out(output.squeeze(1))
        return output, hidden

    def forward(self, inputs, hidden=None, force=True, steps=0):
        if force or steps == 0: steps = len(inputs)
        outputs = Variable(torch.zeros(steps, 1, 1))
        for i in range(steps):
            if force or i == 0:
                input = inputs[i]
            else:
                input = output
            output, hidden = self.step(input, hidden)
            outputs[i] = output
        return outputs, hidden