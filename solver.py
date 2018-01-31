import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from torch.autograd import grad
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms
from model import *
from PIL import Image
from data_loader import PascalVOC2012

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


class Solver(object):
    DEFAULTS = {}
    def __init__(self, data_loader, config):
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.data_loader = data_loader
        # Build tensorboard if use
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def build_model(self):
        # Define a generator and a discriminator
        self.FCN8 = FCN8s(n_class=self.num_classes)
        self.guidance_module = VGG16Modified(n_classes=self.num_classes)
        # Optimizers
        self.optimizer = torch.optim.Adam(self.guidance_module.parameters(), self.lr)

        # Print networks
        self.print_network(self.guidance_module, 'Guidance Network')

        if torch.cuda.is_available():
            self.FCN8.cuda()
            self.guidance_module.cuda()
    
        model_data = torch.load(self.fcn_model_path)
        self.FCN8.load_state_dict(model_data)
        self.FCN8.eval()
    
        self.guidance_module.copy_params_from_vgg16(self.vgg_model_path)
            

    def train(self):
        # The number of iterations per epoch
        print("start training")
        iters_per_epoch = len(self.data_loader)

        fixed_x = []
        fixed_target = []
        for i, (images, target) in enumerate(self.data_loader):
            fixed_x.append(images)
            fixed_target.append(target)
            if i == 1:
                break
        print("sample data")
        # Fixed inputs and target domain labels for debugging
        fixed_x = torch.cat(fixed_x, dim=0)
        fixed_x = self.to_var(fixed_x, volatile=True)
        fixed_target = torch.cat(fixed_target, dim=0)
        # lr cache for decaying
        lr = self.lr

        # Start with trained model if exists
        if self.pretrained_model:
            start = int(self.pretrained_model.split('_')[0])
        else:
            start = 0

        # Start training
        start_time = time.time()

        criterion = CrossEntropyLoss2d(size_average=False, ignore_index=-1).cuda()
        for e in range(start, self.num_epochs):

            for i, (images, target) in enumerate(self.data_loader):
                N = images.size(0)

                # Convert tensor to variable
                images = self.to_var(images)
                target = self.to_var(target)

                coarse_map = self.FCN8(images)

                refined_map= self.guidance_module(images,coarse_map)

                # tmp = refined_map.data.cpu().numpy()

                # assert refined_map.size()[2:] == target.size()[1:]
                # assert refined_map.size()[1] == self.num_classes
                softmax_ce_loss = criterion(refined_map, target) / N

                self.reset_grad()
                softmax_ce_loss.backward()

                self.optimizer.step()


                # # Compute classification accuracy of the discriminator
                # if (i+1) % self.log_step == 0:
                #     accuracies = self.compute_accuracy(real_feature, real_label, self.dataset)
                #     log = ["{:.2f}".format(acc) for acc in accuracies.data.cpu().numpy()]
                #     if self.dataset == 'CelebA':
                #         print('Classification Acc (Black/Blond/Brown/Gender/Aged): ', end='')
                #     else:
                #         print('Classification Acc (8 emotional expressions): ', end='')
                #     print(log)


                # Logging
                loss = {}
                loss['loss'] = softmax_ce_loss.data[0]

                # Print out log info
                if (i+1) % self.log_step == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
                        elapsed, e+1, self.num_epochs, i+1, iters_per_epoch)

                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, e * iters_per_epoch + i + 1)

                # Translate fixed images for debugging
                if (i+1) % self.sample_step == 0:
                    fake_image_list = [torch.from_numpy(self.data_loader.dataset.untransform_batch(fixed_x.data.cpu()))]
                    coarse_map = self.FCN8(fixed_x)
                    refined_map= self.guidance_module(fixed_x,coarse_map)

                    lbl_pred = coarse_map.data.max(1)[1].cpu().numpy()
                    lbl_pred_refined = refined_map.data.max(1)[1].cpu().numpy()
                    lbl_pred = self.data_loader.dataset.colorize_mask_batch(lbl_pred)
                    lbl_pred_refined = self.data_loader.dataset.colorize_mask_batch(lbl_pred_refined)
                    lbl_true = self.data_loader.dataset.colorize_mask_batch(fixed_target.numpy())
                    # print(lbl_pred.size()) 
                    # print(lbl_pred_refined.size()) 
                    # print(lbl_true.size()) 
                    fake_image_list.append(lbl_pred)
                    fake_image_list.append(lbl_pred_refined)
                    fake_image_list.append(lbl_true)
                    # fake_image_list.append(lbl_pred_refined.unsqueeze(1).expand(fixed_x.size()).float())
                    # fake_image_list.append(lbl_true)
                    fake_images = torch.cat(fake_image_list, dim=3)
                    save_image(fake_images,
                        os.path.join(self.sample_path, '{}_{}_fake.png'.format(e+1, i+1)),nrow=1, padding=0)
                    print('Translated images and saved into {}..!'.format(self.sample_path))

                    del coarse_map, refined_map, lbl_pred, lbl_pred_refined, fake_image_list 

                # Save model checkpoints
                if (i+1) % self.model_save_step == 0:
                    torch.save(self.guidance_module.state_dict(),
                        os.path.join(self.model_save_path, '{}_{}_spatial.pth'.format(e+1, i+1)))

            # Decay learning rate
            if (e+1) > (self.num_epochs - self.num_epochs_decay):
                lr -= (self.lr / float(self.num_epochs_decay))
                self.update_lr(lr)
                print ('Decay learning rate to lr: {}.'.format(lr))
    def labels_to_rgb(self,labels):
        return 
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def load_pretrained_model(self):
        self.guidance_module.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_spatial.pth'.format(self.pretrained_model))))

        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        self.optimizer.zero_grad()

    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def threshold(self, x):
        x = x.clone()
        x[x >= 0.5] = 1
        x[x < 0.5] = 0
        return x

    def compute_accuracy(self, x, y, dataset):
        if dataset == 'CelebA':
            x = F.sigmoid(x)
            predicted = self.threshold(x)
            correct = (predicted == y).float()
            accuracy = torch.mean(correct, dim=0) * 100.0
        elif dataset == 'Flowers':
            x = F.sigmoid(x)
            predicted = self.threshold(x)
            correct = (predicted == y).float()
            accuracy = torch.mean(correct, dim=0) * 100.0

        else:
            _, predicted = torch.max(x, dim=1)
            correct = (predicted == y).float()
            accuracy = torch.mean(correct) * 100.0
        return accuracy

    def one_hot(self, labels, dim):
        """Convert label indices to one-hot vector"""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def test(self):
        """Facial attribute transfer on CelebA or facial expression synthesis on RaFD."""
        # Load trained parameters
        G_path = os.path.join(self.model_save_path, '{}_G.pth'.format(self.test_model))
        self.G.load_state_dict(torch.load(G_path))
        self.G.eval()

        if self.dataset == 'CelebA':
            data_loader = self.celebA_loader
        else:
            data_loader = self.rafd_loader

        for i, (real_x, org_c) in enumerate(data_loader):
            real_x = self.to_var(real_x, volatile=True)

            if self.dataset == 'CelebA':
                target_c_list = self.make_celeb_labels(org_c)
            else:
                target_c_list = []
                for j in range(self.c_dim):
                    target_c = self.one_hot(torch.ones(real_x.size(0)) * j, self.c_dim)
                    target_c_list.append(self.to_var(target_c, volatile=True))

            # Start translations
            fake_image_list = [real_x]
            for target_c in target_c_list:
                fake_image_list.append(self.G(real_x, target_c))
            fake_images = torch.cat(fake_image_list, dim=3)
            save_path = os.path.join(self.result_path, '{}_fake.png'.format(i+1))
            save_image(self.denorm(fake_images.data), save_path, nrow=1, padding=0)
            print('Translated test images and saved into "{}"..!'.format(save_path))

   