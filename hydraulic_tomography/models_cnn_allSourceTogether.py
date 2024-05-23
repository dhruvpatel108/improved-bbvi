import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
# ==============================================================================
# Real Box dataset (40x80 ---> 47) CNN model
# ==============================================================================
class CNN_Surrogate(nn.Module):
    def __init__(self, dim):
        super(CNN_Surrogate, self).__init__()

        self.dim = dim
        self.image_to_features = nn.Sequential(
            nn.Conv2d(1, self.dim, (5, 9), (1, 1), (0, 0)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(self.dim),
            nn.Conv2d(self.dim, 2 * self.dim, (5, 9), (1, 1), (0, 0)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(2 * self.dim),
            nn.Conv2d(2 * self.dim, 4 * self.dim, (4, 4), (2, 2), (1, 1)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(4 * self.dim),
            nn.Conv2d(4 * self.dim, 8 * self.dim, (4, 4), (2, 2), (1, 1)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(8 * self.dim),
            nn.Conv2d(8 * self.dim, 16 * self.dim, (4, 4), (2, 2), (1, 1)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(16 * self.dim),
            nn.Conv2d(16 * self.dim, 16 * self.dim, (4, 4), (2, 2), (1, 1)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(16 * self.dim),
        )


        output_size = 16 * self.dim * 2 * 4 
        self.features_to_output = nn.Sequential(
            nn.Linear(output_size, output_size),
            nn.LeakyReLU(0.2),
            nn.Linear(output_size, 47*9)
        )

    def forward(self, input_data):
        batch_size = input_data.size()[0]
        x = self.image_to_features(input_data)
        x = x.reshape(batch_size, -1)
        output = self.features_to_output(x)
        return output

