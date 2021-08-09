# ARS_tfv2

## Requirements

- pybullet
- tensorflow v2.5
- ray
- gym

## Usage

```
python ARS.py
```

## Config
You can change config.py to fit your own flags.

```
hdims             # dimension of hidden layers
nu                # standard deviation of noise
actv = 'tanh'     # activation function
out_actv = 'tanh' # activation function

# ray
n_cpu = n_workers # number of cpu
b = (n_workers//5)# number of top-performing directions to use

# Update
total_steps
ep_len_rollout
alpha             # step size

# Evaluate
max_ep_len_eval
num_eval
evaluate_every
print_every

```
