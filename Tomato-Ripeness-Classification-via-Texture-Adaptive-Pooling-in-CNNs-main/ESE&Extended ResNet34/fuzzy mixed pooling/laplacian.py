import torch
import torch.nn as nn

# Laplacian Variance Layer-------------------------------------------------------------------------------------------------------
class LaplacianVarianceLayer(nn.Module):
    def __init__(self):
        super(LaplacianVarianceLayer, self).__init__()
        self.variances = None
    
    def forward(self, x):
        # Convert the input to grayscale by averaging across the channel dimension
        gray = x.mean(dim=1, keepdim=True)

        # Define a Laplacian kernel
        laplacian_kernel = torch.tensor([[0, 1, 0], 
                                         [1, -8, 1], 
                                         [0, 1, 0]], dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)

        # Apply the Laplacian filter to the grayscale image
        laplacian = torch.nn.functional.conv2d(gray, laplacian_kernel, padding=1)

        # Calculate variance along the (2, 3) dimensions (height, width)
        self.variances = laplacian.var(dim=(2, 3)).cpu().detach().numpy() #-----------------------------------
        #print(self.variances)

                
        return x
