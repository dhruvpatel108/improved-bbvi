import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Generator(nn.Module):
    def __init__(self, img_size, latent_dim, dim):
        super(Generator, self).__init__()

        self.dim = dim
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.feature_sizes = (2, 4)#(int(self.img_size[0] / 16), int(self.img_size[1] / 16))

        self.latent_to_features = nn.Sequential(
            nn.Linear(latent_dim, 16 * dim * self.feature_sizes[0] * self.feature_sizes[1]),
            nn.ReLU()
        )

        self.features_to_image = nn.Sequential(
            nn.ConvTranspose2d(16 * dim, 16 * dim, (4, 4), (2, 2), (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(16 * dim),
            nn.ConvTranspose2d(16 * dim, 8 * dim, (4, 4), (2, 2), (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(8 * dim),
            nn.ConvTranspose2d(8 * dim, 4 * dim, (4, 4), (2, 2), (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(4 * dim),
            nn.ConvTranspose2d(4 * dim, 2 * dim, (4, 4), (2, 2), (1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(2 * dim),
            nn.ConvTranspose2d(2 * dim, dim, (5, 9), (1, 1), (0, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            nn.ConvTranspose2d(dim, self.img_size[2], (5, 9), (1, 1), (0, 0)),
            nn.Tanh()
        )

    def forward(self, input_data):
        # Map latent into appropriate size for transposed convolutions
        x = self.latent_to_features(input_data)
        # Reshape
        x = x.view(-1, 16 * self.dim, self.feature_sizes[0], self.feature_sizes[1])
        # Return generated image
        output = self.features_to_image(x)
        return output

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.latent_dim))