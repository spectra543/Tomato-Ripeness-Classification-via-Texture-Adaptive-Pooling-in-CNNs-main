import torch
import torch.nn as nn
import numpy as np
import csv
import os

class LaplacianVarianceLayer(nn.Module):
    def __init__(self, filename='var_restnet34.csv'):
        super(LaplacianVarianceLayer, self).__init__()
        self.filename = filename
        self.variances = None

        # Check if file exists, and create with header if not
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Variance'])  # Header

    def forward(self, x):
        # Convert to grayscale
        gray = x.mean(dim=1, keepdim=True)

        # Laplacian kernel
        laplacian_kernel = torch.tensor([[0, 1, 0], 
                                         [1, -8, 1], 
                                         [0, 1, 0]], dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)

        # Apply Laplacian filter
        laplacian = torch.nn.functional.conv2d(gray, laplacian_kernel, padding=1)

        # Compute variance across spatial dims
        self.variances = laplacian.var(dim=(2, 3)).cpu().detach().numpy()  # shape: [batch_size, 1]

        # Append to CSV immediately
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.variances)

        return x

