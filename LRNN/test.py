from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from data_loader import *
import matplotlib.pyplot as plt
from model import *
from torch.autograd import Variable

if __name__ == '__main__':
    voc_root = "E:/Jonathan"
    data_loader = get_loader(voc_root, 96,
                                 96, 1, 'PascalVOC2012', 'train')

    a = np.random.randn(20,3,96,96)
    a = Variable(torch.from_numpy(a).float()).cuda()

    gen = LRNN().cuda()

    gen(a)
    
##    for batch_idx, (data, target) in enumerate(data_loader):
##
##        print(data.size())
##            
##            plt.subplot(131)
##            plt.imshow(img)
##            plt.subplot(132)
##            plt.imshow(lp)
##            plt.subplot(133)
##            plt.imshow(lt)
##            plt.show()
##        break
