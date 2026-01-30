import torch
import torch.nn as nn
import torch.nn.functional as F

print("alpha = 0.3")

class mixedPool(nn.Module):
    def __init__(self, alpha=0.3, padding=0):
        # nn.Module.__init__(self)
        super(mixedPool, self).__init__()
        alpha = torch.FloatTensor([alpha])
        self.alpha = nn.Parameter(alpha)  # nn.Parameter is special Variable
        self.padding = padding

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        k_h, k_w = height, width
        #print(f"Current alpha value: {self.alpha.item()}")
        x = self.alpha * F.max_pool2d(x, kernel_size=(k_h, k_w), stride=(k_h, k_w), padding=self.padding) + (
                    1 - self.alpha) * -F.max_pool2d(-x, kernel_size=(k_h, k_w), stride=(k_h, k_w), padding=self.padding)
        return x

def mixed():
    #print("You are using Mixed Pooling Method")
    return mixedPool

