from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from voc import *
import matplotlib.pyplot as plt
from model import *
from torch.autograd import Variable
import time

if __name__ == '__main__':
    voc_root = "E:/Jonathan"
    data_loader = DataLoader(VOC2012ClassSeg(voc_root, split='train', transform=True),
        batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True)


    model = FCN8s(n_class=21)

    guidance_module = VGG16Modified()
    if torch.cuda.is_available():
        model = model.cuda()
        guidance_module = guidance_module.cuda()

    model_file = "./model_weights/fcn8s_from_caffe.pth"
    vgg_model_file = "./model_weights/vgg16_from_caffe.pth"

    
    model_data = torch.load(model_file)
    model.load_state_dict(model_data)
##    model.eval()
    
    guidance_module.copy_params_from_vgg16(vgg_model_file)

    
    for batch_idx, (data, target) in enumerate(data_loader):
        t_num = target.numpy()
        print(np.sum(t_num==0) / (128*128))
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data, volatile=True), Variable(target)
        st = time.time()
        coarse_map = model(data)
        refined = guidance_module(data,coarse_map)
        print(refined.size(), time.time()-st)


        break


        
##        lbl_pred = score.data
##        print(lbl_pred.size())
##        break
##        imgs = data.data.cpu()
##        lbl_pred = coarse_map.data.max(1)[1].cpu().numpy()[:, :, :]
##        lbl_true = target.data.cpu()
##        for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
##            img, lt = data_loader.dataset.untransform(img, lt)
##            print(img.shape)
##            print(lt.shape)
##            print(lp.shape)
##            plt.subplot(131)
##            plt.imshow(img)
##            plt.subplot(132)
##            plt.imshow(lp)
##            plt.subplot(133)
##            plt.imshow(lt)
##            plt.show()


