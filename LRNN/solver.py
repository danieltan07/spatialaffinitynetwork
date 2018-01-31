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
        self.LRNN = LRNN()

        # Optimizers
        self.optimizer = torch.optim.Adam(self.LRNN.parameters(), self.lr)
      
        # Print networks
        self.print_network(self.LRNN, 'LRNN')

        if torch.cuda.is_available():
            self.LRNN.cuda()


    def train(self):
        """Train StarGAN within a single dataset."""
        loss_fn = torch.nn.MSELoss(size_average=False)
        # The number of iterations per epoch
        iters_per_epoch = len(self.data_loader)

        fixed_x = []
        real_c = []
        for i, (images, target) in enumerate(self.data_loader):
            fixed_x.append(images)
            if i == 16:
                break

        # Fixed inputs and target domain labels for debugging
        fixed_x = torch.cat(fixed_x, dim=0)
        fixed_x = self.to_var(fixed_x, volatile=True)

        # lr cache for decaying
        lr = self.lr

        # Start with trained model if exists
        if self.pretrained_model:
            start = int(self.pretrained_model.split('_')[0])
        else:
            start = 0

        # Start training
        start_time = time.time()
        for e in range(start, self.num_epochs):
            for i, (images, target) in enumerate(self.data_loader):
                
                # Convert tensor to variable
                images = self.to_var(images)
                target = self.to_var(target)

                refined_images = self.LRNN(images)

                mse_loss = loss_fn(refined_images, target) / images.size(0)

                self.reset_grad()
                mse_loss.backward()
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
                loss['loss'] = mse_loss.data[0]
                
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
                    fake_image_list = [fixed_x]
                    refined_images = self.LRNN(fixed_x)
                    fake_image_list.append(refined_images)
                    fake_images = torch.cat(fake_image_list, dim=3)
                    save_image(fake_images.data,
                        os.path.join(self.sample_path, '{}_{}_fake.png'.format(e+1, i+1)),nrow=1, padding=0)
                    print('Translated images and saved into {}..!'.format(self.sample_path))

                # Save model checkpoints
                if (i+1) % self.model_save_step == 0:
                    torch.save(self.LRNN.state_dict(),
                        os.path.join(self.model_save_path, '{}_{}_LRNN.pth'.format(e+1, i+1)))

            # Decay learning rate
            if (e+1) > (self.num_epochs - self.num_epochs_decay):
                lr -= (self.lr / float(self.num_epochs_decay))
                self.update_lr(lr)
                print ('Decay learning rate to lr: {}.'.format(lr))

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def load_pretrained_model(self):
        self.LRNN.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_LRNN.pth'.format(self.pretrained_model))))
        # self.D.load_state_dict(torch.load(os.path.join(
        #     self.model_save_path, '{}_D.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        # for param_group in self.d_optimizer.param_groups:
        #     param_group['lr'] = d_lr

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

    def make_celeb_labels(self, real_c):
        """Generate domain labels for CelebA for debugging/testing.

        if dataset == 'CelebA':
            return single and multiple attribute changes
        elif dataset == 'Both':
            return single attribute changes
        """
        y = [torch.FloatTensor([1, 0, 0]),  # black hair
             torch.FloatTensor([0, 1, 0]),  # blond hair
             torch.FloatTensor([0, 0, 1])]  # brown hair

        fixed_c_list = []

        # single attribute transfer
        for i in range(self.c_dim):
            fixed_c = real_c.clone()
            for c in fixed_c:
                if i < 3:
                    c[:3] = y[i]
                else:
                    c[i] = 0 if c[i] == 1 else 1   # opposite value
            fixed_c_list.append(self.to_var(fixed_c, volatile=True))

        # multi-attribute transfer (H+G, H+A, G+A, H+G+A)
        if self.dataset == 'CelebA':
            for i in range(4):
                fixed_c = real_c.clone()
                for c in fixed_c:
                    if i in [0, 1, 3]:   # Hair color to brown
                        c[:3] = y[2] 
                    if i in [0, 2, 3]:   # Gender
                        c[3] = 0 if c[3] == 1 else 1
                    if i in [1, 2, 3]:   # Aged
                        c[4] = 0 if c[4] == 1 else 1
                fixed_c_list.append(self.to_var(fixed_c, volatile=True))
        return fixed_c_list

    def make_flowers_labels(self,real_c):
        """Generate domain labels for CelebA for debugging/testing.

        if dataset == 'CelebA':
            return single and multiple attribute changes
        elif dataset == 'Both':
            return single attribute changes
        """

        fixed_c_list = []

        # single attribute transfer
        for i in range(self.c_dim):
            fixed_c = real_c.clone()
            for c in fixed_c:
                c[:] = torch.FloatTensor(np.eye(self.c_dim)[i])
            fixed_c_list.append(self.to_var(fixed_c, volatile=True))

        return fixed_c_list
   
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

    def test_multi(self):
        """Facial attribute transfer and expression synthesis on CelebA."""
        # Load trained parameters
        G_path = os.path.join(self.model_save_path, '{}_G.pth'.format(self.test_model))
        self.G.load_state_dict(torch.load(G_path))
        self.G.eval()

        for i, (real_x, org_c) in enumerate(self.celebA_loader):

            # Prepare input images and target domain labels
            real_x = self.to_var(real_x, volatile=True)
            target_c1_list = self.make_celeb_labels(org_c)
            target_c2_list = []
            for j in range(self.c2_dim):
                target_c = self.one_hot(torch.ones(real_x.size(0)) * j, self.c2_dim)
                target_c2_list.append(self.to_var(target_c, volatile=True))

            # Zero vectors and mask vectors
            zero1 = self.to_var(torch.zeros(real_x.size(0), self.c2_dim))     # zero vector for rafd expressions
            mask1 = self.to_var(self.one_hot(torch.zeros(real_x.size(0)), 2)) # mask vector: [1, 0]
            zero2 = self.to_var(torch.zeros(real_x.size(0), self.c_dim))      # zero vector for celebA attributes
            mask2 = self.to_var(self.one_hot(torch.ones(real_x.size(0)), 2))  # mask vector: [0, 1]

            # Changing hair color, gender, and age
            fake_image_list = [real_x]
            for j in range(self.c_dim):
                target_c = torch.cat([target_c1_list[j], zero1, mask1], dim=1)
                fake_image_list.append(self.G(real_x, target_c))

            # Changing emotional expressions
            for j in range(self.c2_dim):
                target_c = torch.cat([zero2, target_c2_list[j], mask2], dim=1)
                fake_image_list.append(self.G(real_x, target_c))
            fake_images = torch.cat(fake_image_list, dim=3)

            # Save the translated images
            save_path = os.path.join(self.result_path, '{}_fake.png'.format(i+1))
            save_image(self.denorm(fake_images.data), save_path, nrow=1, padding=0)
            print('Translated test images and saved into "{}"..!'.format(save_path))