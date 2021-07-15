# Configuration
n_cpu = n_workers = 50
total_steps = 5000
evaluate_every = 10
print_every = 10
ep_len_rollout = 1000   #1000
num_eval = 3
max_ep_len_eval = 1e3   #1e3
hdims = [256,256]
increment = -50   # 목표 ret 증가량
actv = 'tanh'
out_actv = 'tanh'

alpha = 0.01    #learning_rate
nu = 0.06
b = (n_workers//5)      # 0.01,0.05,(n_workers//5)