import torch.nn as nn

# Configuration
n_cpu = n_workers = 10
total_steps = 5000
evaluate_every = 10 #50
print_every = 10
ep_len_rollout = 1000   #1000
num_eval = 3
max_ep_len_eval = 1e3   #1e3
hdims = [32,16]
actv = 'tanh'
out_actv = 'tanh'

alpha = 0.01    #learning_rate
nu = 0.05
b = (n_workers//5)      # 0.01,0.05,(n_workers//5)