'''
Uses pretrained VAE to process dataset to get mu and logvar for each frame, and stores
all the dataset files into one dataset called series/series.npz
'''

import numpy as np
import os
import json
import tensorflow as tf
import random
from vae.vae_torch import *
from constants import *

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
num_episode_to_use = 10000
DATA_DIR = "./record"
SERIES_DIR = "./series"
model_path_name = "./tf_vae"
filesize=1000

if not os.path.exists(SERIES_DIR):
    os.makedirs(SERIES_DIR)

def load_raw_data_list(filelist):
  data_list = []
  action_list = []
  counter = 0
  for i in range(len(filelist)):
    filename = filelist[i]
    raw_data = np.load(os.path.join(DATA_DIR, filename))
    data_list.append(raw_data['obs']) #1000x64X64x3
    action_list.append(raw_data['action']) #1000x64X64x3
    if ((i+1) % 1000 == 0):
      print("loading file", (i+1))
  return data_list, action_list

def encode_batch_tf(batch_img):
  simple_obs = np.copy(batch_img).astype(np.float)/255.0
  simple_obs = simple_obs.reshape(batch_size, 3, 64, 64)
  mu, logvar = vae.encode_mu_logvar(simple_obs) #1000x32
  z = (mu + np.exp(logvar/2.0) * np.random.randn(*logvar.shape))
  return mu, logvar, z

def encode_batch(batch_img, device):
    simple_obs = np.copy(batch_img).astype(np.float) / 255.0
    simple_obs = simple_obs.reshape(batch_size, 3, 64, 64)
    simple_obs = torch.tensor(simple_obs, device=device, dtype=torch.float)
    _, mu, logvar = vae(simple_obs)  # 1000x32
    mu = mu.cpu().detach().numpy()
    logvar = logvar.cpu().detach().numpy()
    z = (mu + np.exp(logvar / 2.0) * np.random.randn(*logvar.shape))
    return mu, logvar, z

def decode_batch(batch_z):
  # decode the latent vector
  batch_img = vae.decode(z.reshape(batch_size, z_size)) * 255.
  batch_img = np.round(batch_img).astype(np.uint8)
  batch_img = batch_img.reshape(batch_size, 64, 64, 3)
  return batch_img


# Hyperparameters for ConvVAE
z_size=32
batch_size=1000 # treat every episode as a batch of 1000!
learning_rate=0.0001
kl_tolerance=0.5

filelist = os.listdir(DATA_DIR)
filelist.sort()
filelist = filelist[0:num_episode_to_use]

file_lists_chunks = [filelist[i * filesize:(i + 1) * filesize] for i in range((len(filelist) + filesize - 1) // filesize)]

#dataset, action_dataset = load_raw_data_list(filelist) #dataset has num_episode_to_use entries, each with size 1000x64X64x3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE(z_size=z_size, kl_tolerance=kl_tolerance, learning_rate=vae_learning_rate)
vae.load_state_dict(torch.load("tf_vae/vae.pt", map_location=device))
vae.to(device)
vae = vae.eval()


mu_dataset = []
logvar_dataset = []
action_dataset = []
for filelists in file_lists_chunks:
  dataset, actions = load_raw_data_list(filelists)
  action_dataset.extend(actions)
  for i in range(len(dataset)):
      data_batch = dataset[i] #1000x64X64x3
      mu, logvar, z = encode_batch(data_batch, device) #all three entries have size 1000x32
      mu_dataset.append(mu.astype(np.float16))
      logvar_dataset.append(logvar.astype(np.float16))
      if ((i+1) % 100 == 0):
        print(i+1)

action_dataset = np.array(action_dataset)
mu_dataset = np.array(mu_dataset)
logvar_dataset = np.array(logvar_dataset)

np.savez_compressed(os.path.join(SERIES_DIR, "series.npz"), action=action_dataset, mu=mu_dataset, logvar=logvar_dataset)
