'''
train mdn-rnn from pre-processed data.
also save 1000 initial mu and logvar, for generative experiments (not related to training).
'''

import numpy as np
import os
import json
import tensorflow as tf
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from vae.vae import ConvVAE, reset_graph
from rnn.rnn import HyperParams
from rnn.rnn_torch import MDNRNN, save_model
from constants import *

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
torch.backends.cudnn.enabled = False
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

DATA_DIR = "./series"
model_save_path = "./tf_rnn"
if not os.path.exists(model_save_path):
  os.makedirs(model_save_path)
  
initial_z_save_path = "tf_initial_z"
if not os.path.exists(initial_z_save_path):
  os.makedirs(initial_z_save_path)

def random_batch():
  indices = np.random.permutation(N_data)[0:batch_size]
  mu = data_mu[indices] #100x1000x32
  logvar = data_logvar[indices]
  action = data_action[indices]
  s = logvar.shape
  z = mu + np.exp(logvar/2.0) * np.random.randn(*s)
  return z, action

def default_hps():
  return HyperParams(num_steps=4000,
                     max_seq_len=999, # train on sequences of 1000 (so 999 + teacher forcing shift)
                     input_seq_width=35,    # width of our data (32 + 3 actions)
                     output_seq_width=32,    # width of our data is 32
                     rnn_size=256,    # number of rnn cells
                     batch_size=100,   # minibatch sizes
                     grad_clip=1.0,
                     num_mixture=5,   # number of mixtures in MDN
                     learning_rate=0.001,
                     decay_rate=1.0,
                     min_learning_rate=0.00001,
                     use_layer_norm=0, # set this to 1 to get more stable results (less chance of NaN), but slower
                     use_recurrent_dropout=0,
                     recurrent_dropout_prob=0.90,
                     use_input_dropout=0,
                     input_dropout_prob=0.90,
                     use_output_dropout=0,
                     output_dropout_prob=0.90,
                     is_training=1)

hps_model = default_hps()
hps_sample = hps_model._replace(batch_size=1, max_seq_len=1, use_recurrent_dropout=0, is_training=0)

raw_data = np.load(os.path.join(DATA_DIR, "series.npz"))

# load preprocessed data
data_mu = raw_data["mu"] #num_episodex1000x32
data_logvar = raw_data["logvar"]
data_action =  raw_data["action"]
max_seq_len = hps_model.max_seq_len

N_data = len(data_mu) # should be 10k
batch_size = hps_model.batch_size

# save 1000 initial mu and logvars:
initial_mu = np.copy(data_mu[:1000, 0, :]*10000).astype(np.int).tolist()
initial_logvar = np.copy(data_logvar[:1000, 0, :]*10000).astype(np.int).tolist()
with open(os.path.join("tf_initial_z", "initial_z.json"), 'wt') as outfile:
  json.dump([initial_mu, initial_logvar], outfile, sort_keys=True, indent=0, separators=(',', ': '))


hps = hps_model
device = torch.device('cuda')
#rnn = MDNRNN(hps_model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rnn_model = MDNRNN(rnn_input_size, device, rnn_output_size=rnn_output_size).to(device)
if os.path.exists("tf_rnn/rnn.pt"):
  rnn_model.load_state_dict(torch.load("tf_rnn/rnn.pt", map_location=device))

rnn_model.to(device)
rnn_model = rnn_model.train()
optimizer = optim.Adam(rnn_model.parameters(), lr=hps.learning_rate)

start = time.time()
step = 0
for local_step in range(hps.num_steps):
  step = step + 1
  curr_learning_rate = (hps.learning_rate-hps.min_learning_rate) * (hps.decay_rate) ** step + hps.min_learning_rate

  optimizer.zero_grad()
  raw_z, raw_a = random_batch() # raw_z = 100x1000x32, raw_a= 100x1000x3
  inputs = np.concatenate((raw_z[:, :-1, :], raw_a[:, :-1, :]), axis=2) #100x1000x35
  outputs = raw_z[:, 1:, :] # teacher forcing (shift by one predictions) #100x999x32
  inputs = torch.tensor(inputs, device=device, dtype=torch.float)
  outputs = torch.tensor(outputs, device=device, dtype=torch.float)

  hidden = rnn_model.init_hidden(hps.batch_size) #( (1x100x256), (1x100x256))
  #print(step, inputs.size()) #torch.Size([100, 999, 35])
  inputs = inputs.view(-1, hps.batch_size, rnn_input_size)
  (pi, mu, sigma), (h, c) = rnn_model(inputs, hidden)
  #print("pi", pi.size(), mu.size(), sigma.size()) #pi torch.Size([3196800, 5]) torch.Size([3196800, 5]) torch.Size([3196800, 5])
  loss = rnn_model.get_lossfunc(pi, mu, sigma, outputs)

  loss.backward()
  for p in rnn_model.parameters():
    p.grad.data.clamp(-1.0, 1.0)
  optimizer.step()

  if (step%20==0 and step > 0):
    end = time.time()
    time_taken = end-start
    start = time.time()
    output_log = "step: %d, lr: %.6f, cost: %.4f, train_time_taken: %.4f" % (step, curr_learning_rate, loss.item(), time_taken)
    print(output_log)
  if (step%500 == 0):
    torch.save(rnn_model.state_dict(), "tf_rnn/rnn.pt")

# save the model (don't bother with tf checkpoints json all the way ...)
torch.save(rnn_model.state_dict(), "tf_rnn/rnn.pt")
