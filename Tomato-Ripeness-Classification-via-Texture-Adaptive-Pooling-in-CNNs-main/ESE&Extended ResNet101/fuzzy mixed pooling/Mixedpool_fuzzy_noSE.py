import torch
import torch.nn as nn
import torch.nn.functional as F
from laplacian import LaplacianVarianceLayer
from mamdani_fuzzy_Rest101_noSE import compute_fuzzy_radius
#from RestnetandDarknet_mixfuzz_module.mamdani_fuzzy_Rest34 import compute_fuzzy_radius
#from RestnetandDarknet_mixfuzz_module.mamdani_fuzzy_darknet53 import compute_fuzzy_radius

print("fuzzy_pool_noSE")

class mixedPool_fuzz(nn.Module):
    def __init__(self, padding=0):
        # nn.Module.__init__(self)
        super(mixedPool_fuzz, self).__init__()
        self.padding = padding
        self.laplacian_layer = LaplacianVarianceLayer()

    def forward(self, x):
    
        batch_size, channels, height, width = x.size()
        k_h, k_w = height, width
        
        x = self.laplacian_layer(x)

        alpha_fuzzy_value = compute_fuzzy_radius(self.laplacian_layer.variances)
        alpha_fuzzy_value = torch.tensor(alpha_fuzzy_value, dtype=x.dtype, device=x.device)
        
        alpha_fuzzy_value = alpha_fuzzy_value.view(-1, 1, 1, 1)
        #print(alpha_fuzzy_value)
        x = alpha_fuzzy_value * F.max_pool2d(x, kernel_size=(k_h, k_w), stride=(k_h, k_w), padding=self.padding) + (
                    1 - alpha_fuzzy_value) * -F.max_pool2d(-x, kernel_size=(k_h, k_w), stride=(k_h, k_w), padding=self.padding)
        return x

def mixed():
    #print("You are using Mixed Pooling Method")
    return mixedPool

