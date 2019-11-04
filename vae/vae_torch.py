# ConvVAE model

import numpy as np
import json
import os
from constants import kl_tolerance, z_size
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

#dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=4)
#for i_batch, sample_batched in enumerate(dataloader):
#torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

class VAE(nn.Module):
    def __init__(self, z_size, kl_tolerance, learning_rate=0.0001):
        super(VAE, self).__init__()
        self.z_size = z_size
        self.kl_tolerance = kl_tolerance
        self.learning_rate = learning_rate
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)  # in_channels, out_channels, kernel_size
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)

        self.mu = nn.Linear(2 * 2 * 256, self.z_size)
        self.logvar = nn.Linear(2 * 2 * 256, self.z_size)

        self.deconvfc = nn.Linear(self.z_size, 2*2*256)
        self.deconv1 = nn.ConvTranspose2d(2*2*256, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)

    def encoder(self, x): #[None, 3, 64, 64]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 2 * 2 * 256)
        return self.mu(x), self.logvar(x)

    def reparameterize(self, mu, logvar):
        sigma = torch.exp(logvar / 2.0)
        epsilon = torch.randn_like(mu)
        return mu + sigma*epsilon
    def decoder(self, z):
        z = self.deconvfc(z)
        z = z.view(-1, 4 * 256, 1, 1)
        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))
        z = F.relu(self.deconv3(z))
        z = torch.sigmoid(self.deconv4(z))
        return z
    def forward(self, x):
        #print(x.size())
        mu, logvar = self.encoder(x) #self.encoder(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        #print(z.size())
        return self.decoder(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    r_loss = torch.sum((x-recon_x).pow(2), dim=(1, 2, 3))
    #print(x-recon_x)
    r_loss = r_loss.mean()
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = torch.max(kl_loss, kl_loss.new([kl_tolerance * z_size]))
    kl_loss = kl_loss.mean()
    #print(r_loss, kl_loss)
    return r_loss + kl_loss, r_loss, kl_loss

def save_model(path, vae):
        torch.save({
            'model_state_dict': vae.VAE_model.state_dict()
        }, path)


def load_model(path, vae, device):
    checkpoint = torch.load(path, map_location=device)
    vae.load_state_dict(checkpoint)
    return vae

class ConvVAE(object):
    def __init__(self, z_size=32, batch_size=1, learning_rate=0.0001, kl_tolerance=0.5, is_training=False, reuse=False,
                 gpu_mode=False):
        self.z_size = z_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.is_training = is_training
        self.kl_tolerance = kl_tolerance
        self.reuse = reuse
        self._build_graph()
    def _build_graph(self):
        #self.x = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.VAE_model = VAE(self.z_size, self.kl_tolerance).to(self.device)
        self.optimizer = optim.Adam(self.VAE_model.parameters(), lr=self.learning_rate)