# Spatial Affinity Networks
PyTorch Implementation of the paper Learning Affinity via Spatial Propagation Networks

[Work in Progress]

FCN-8 Code and Model adopted from https://github.com/wkentaro/pytorch-fcn

The paper propose to learn a spatial affinity matrix by consturcting a row-wise / column-wise linear propagation model where each pixel in the current row/column incorporates the information from its three adjacent pixels in the previous row/column. The idea is to train a CNN that generates the weights of a recursive filter conditioned on an input image. Doing this for all 4 directions, you can form global and densely connected pairwise relations, as shown in Figure 1.


![alt text](https://github.com/danieltan07/spatialaffinitynetwork/blob/master/fig1.PNG)


As shown in Figure 2, the model is separated into two modules: (1) a guidance network that outputs the weights / elements of the transformation matrix; (2) a propagation module that uses the weights given by the guidance network and uses it as a recursive filter to refine the result. 

![alt text](https://github.com/danieltan07/spatialaffinitynetwork/blob/master/fig2.PNG)

## Some Implementation Details

The guidance network outputs a tensor of size H x W x (C x 3 weights x 4 directions). We need to convert the tensor to a tridiagonal matrix so that when we perform a dot product it will correspond to the weights of the three adjacent pixels in the previous row/column. 

```python
    def to_tridiagonal_multidim(self, w):
        # this function converts the weight vectors to a tridiagonal matrix
        
        N,W,C,D = w.size()
        
        # normalize the weights to stabilize the model
        tmp_w = w / torch.sum(torch.abs(w),dim=3).unsqueeze(-1)
        tmp_w = tmp_w.unsqueeze(2).expand([N,W,W,C,D])
        
        # three identity matrices, one normal, one shifted left and the other shifted right
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
```
