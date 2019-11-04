'''
Train VAE model on data created using extract.py
final model saved into tf_vae/vae.json
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1" # can just override for multi-gpu systems
import time
import tensorflow as tf
import random
import numpy as np
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)
import torch
from vae.vae_torch import VAE, loss_function, save_model, load_model
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from constants import *

# Hyperparameters for ConvVAE
z_size=32
batch_size=100
learning_rate=0.0001
kl_tolerance=0.5
num_episode_to_use = 10000
# Parameters for training
NUM_EPOCH = 10
DATA_DIR = "./record"
filesize=1000

model_save_path = "./tf_vae"
if not os.path.exists(model_save_path):
  os.makedirs(model_save_path)

def count_length_of_filelist(filelist):
  # although this is inefficient, much faster than doing np.concatenate([giant list of blobs])..
  N = len(filelist)
  total_length = 0
  for i in range(N):
    filename = filelist[i]
    raw_data = np.load(os.path.join("record", filename))['obs']
    l = len(raw_data)
    total_length += l
    if (i % 1000 == 0):
      print("loading file", i)
  return  total_length

def create_dataset(filelist, N=10000, M=1000): # N is 10000 episodes, M is number of timesteps
  #print(N)
  data = np.zeros((M*N, 64, 64, 3), dtype=np.uint8)
  idx = 0
  for i in range(N):
    filename = filelist[i]
    raw_data = np.load(os.path.join("record", filename))['obs']
    l = len(raw_data)
    if (idx+l) > (M*N):
      data = data[0:idx]
      print('premature break')
      break
    data[idx:idx+l] = raw_data
    idx += l
    #if ((i+1) % 100 == 0):
    #  print("loading file", i+1)
  #data = np.moveaxis(data, 3, 1) #from (batch, 64, 64, 3) to (batch, 3, 64, 64)
  return data

# load dataset from record/*. only use first 10K, sorted by filename.
filelist = os.listdir(DATA_DIR)
filelist.sort()
filelist = filelist[0:num_episode_to_use]
#print("check total number of images:", count_length_of_filelist(filelist))
file_lists_chunks = [filelist[i * filesize:(i + 1) * filesize] for i in range((len(filelist) + filesize - 1) // filesize)]
#dataset = create_dataset(filelist, N=num_episode_to_use)

# split into batches:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE(z_size=z_size, kl_tolerance=kl_tolerance, learning_rate=vae_learning_rate)
vae.to(device)
vae_model = vae.train()
optimizer = optim.Adam(vae_model.parameters(), lr=vae_model.learning_rate)

train_step = 0
print("train", "step", "loss", "recon_loss", "kl_loss")


for epoch in range(NUM_EPOCH):
  start = time.time()
  for i, file_list in enumerate(file_lists_chunks):
    dataset = create_dataset(file_list, len(file_list), 1000)
    total_length = len(dataset)
    num_batches = int(np.floor(total_length / batch_size)) #1
    np.random.shuffle(dataset)
    #dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    for idx in range(num_batches):
    #for i_batch, batch in enumerate(dataloader):
      batch = dataset[idx*(batch_size):(idx+1)*batch_size] #dataset[(idx+1)*(1):(idx+2)*1] #
      batch = torch.tensor(batch, dtype=torch.float, device=device)/255.0
      batch = batch.view(-1, 3, 64, 64)
      optimizer.zero_grad()
      recons_batch, mu, logvar = vae_model(batch)
      loss, r_loss, kl_loss = loss_function(recons_batch, batch, mu, logvar)
      #print(loss.item())
      loss.backward()
      optimizer.step()
      if (train_step + 1) % 500 == 0:
        print("step", (train_step + 1), loss.item(), r_loss.item(), kl_loss.item())
      if (train_step + 1) % 5000 == 0:
        #save_model("tf_vae/vae.pt", vae)
        torch.save(vae_model.state_dict(), "tf_vae/vae.pt")
      train_step += 1
  #end = time.time()
#save_model("./tf_vae/vae.pt", vae)
torch.save(vae_model.state_dict(), "tf_vae/vae.pt")
'''train_step = 0
for epoch in range(1000):

    batch = torch.tensor(single_image, dtype=torch.float, device=device)/255.0
    batch=  batch.view(-1, 3, 64, 64)
    #print(batch.size())
    optimizer.zero_grad()
    recons_batch, mu, logvar = vae_model(batch)
    if (epoch+1)%200 == 0:
      print(recons_batch)
    loss, r_loss, kl_loss = loss_function(recons_batch, batch, mu, logvar)
    #print(loss.item())
    loss.backward()
    optimizer.step()
    print("step", (train_step + 1), loss.item(), r_loss.item(), kl_loss.item())
    if (epoch+1)%100 == 0:
      torch.save(vae_model.state_dict(), "tf_vae/vae.pt")
    train_step += 1
torch.save(vae_model.state_dict(), "tf_vae/vae.pt")'''
