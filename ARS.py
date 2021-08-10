import datetime,gym,os,pybullet_envs,time,os,psutil,ray
import numpy as np
import tensorflow as tf
from model import MLP, get_noises_from_weights
from collections import deque
# from util import gpu_sess,suppress_tf_warning
np.set_printoptions(precision=2)
# suppress_tf_warning() # suppress warning  
gym.logger.set_level(40) # gym logger
print ("Packaged loaded. TF version is [%s]."%(tf.__version__))

RENDER_ON_EVAL = False

class RolloutWorkerClass(object):
    """
    Worker without RAY (for update purposes)
    """
    def __init__(self, args, seed=1):
        self.seed = seed
        self.env = get_env()
        odim, adim = self.env.observation_space.shape[0], self.env.action_space.shape[0]
        self.odim = odim
        self.adim = adim
        # ARS Model
        self.model = MLP(odim, adim, hdims=args.hdims, actv=args.actv, out_actv=args.out_actv)
        # Initialize model
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

    @tf.function
    def get_action(self, o):
        return self.model(o)[0]

    @tf.function
    def get_weights(self):
        """
        Get weights
        """
        weight_vals = self.model.trainable_weights
        return weight_vals

    @tf.function
    def set_weights(self, weight_vals):
        """
        Set weights without memory leakage
        """
        for old_weight, new_weight  in zip(self.model.trainable_weights, weight_vals):
            old_weight.assign(new_weight)

    def save_weight(self, log_path):
        self.model.save_weights(log_path + "/weights/weights")

    def load_weight(self, checkpoint):
        self.model.load_weights(checkpoint)

@ray.remote
class RayRolloutWorkerClass(object):
    """
    Rollout Worker with RAY
    """
    def __init__(self,args,worker_id=0,):
        self.worker_id = worker_id
        self.ep_len_rollout = args.ep_len_rollout
        self.env = get_env()
        odim, adim = self.env.observation_space.shape[0], self.env.action_space.shape[0]
        self.odim = odim
        self.adim = adim
        # ARS Model
        self.model = MLP(odim, adim, hdims=args.hdims, actv=args.actv, out_actv=args.out_actv)

    @tf.function
    def get_action(self, o):
        return self.model(o)[0]

    # @tf.function
    def set_weights(self, weight_vals, noise_vals, noise_sign=+1):
        """
        Set weights without memory leakage
        """
        for idx, weight in enumerate(self.model.trainable_weights):
            weight.assign(weight_vals[idx]+noise_sign*noise_vals[idx])

    def rollout(self):
        """
        Rollout
        """
        # Loop
        self.o = self.env.reset() # reset always
        r_sum,step = 0,0
        for t in range(self.ep_len_rollout):
            self.a = self.get_action(self.o.reshape(1, -1))
            self.o2,self.r,self.d,_ = self.env.step(self.a)
            # Save next state
            self.o = self.o2
            # Accumulate reward
            r_sum += self.r
            step += 1
            if self.d: break
        return r_sum,step

class Agent(object):
    def __init__(self, args, seed=1):
        # Config
        self.n_cpu = self.n_workers = args.n_cpu
        self.total_steps = args.total_steps
        self.evaluate_every = args.evaluate_every
        self.print_every = args.print_every
        self.num_eval = args.num_eval
        self.max_ep_len_eval = args.max_ep_len_eval
        self.alpha = args.alpha
        self.nu = args.nu

        self.seed = seed
        # Environment
        self.env = get_env()
        odim, adim = self.env.observation_space.shape[0],self.env.action_space.shape[0]
        self.odim = odim
        self.adim = adim

        ray.init(num_cpus=self.n_cpu)
        self.R = RolloutWorkerClass(args, seed=0)
        self.workers = [RayRolloutWorkerClass.remote(
            args, worker_id=i,
        ) for i in range(self.n_workers)]

        self.log_path = "./log/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.summary_writer = tf.summary.create_file_writer(self.log_path + "/summary/")

    def train(self, load_dir=None):
        start_time = time.time()
        n_env_step = 0
        eval_env = get_eval_env()
        latest_100_score = deque(maxlen=100)

        # if load_dir:
        #     loaded_ckpt = tf.train.latest_checkpoint(load_dir)
        #     self.model.load_weights(loaded_ckpt)
        #     print('load weights')
        for t in range(int(self.total_steps)):
            # Distribute worker weights
            weights = self.R.get_weights()
            noises_list = []
            for _ in range(self.n_workers):
                noises_list.append(get_noises_from_weights(weights, nu=self.nu))

            # Positive rollouts (noise_sign=+1)
            set_weights_list = [worker.set_weights.remote(weights, noises, noise_sign=1)
                                for worker, noises in zip(self.workers, noises_list)]
            ops = [worker.rollout.remote() for worker in self.workers]
            res_pos = ray.get(ops)
            rollout_pos_vals, r_idx = np.zeros(self.n_workers), 0
            for rew, eplen in res_pos:
                rollout_pos_vals[r_idx] = rew
                r_idx = r_idx + 1
                n_env_step += eplen

            # Negative rollouts (noise_sign=-1)
            set_weights_list = [worker.set_weights.remote(weights, noises, noise_sign=-1)
                                for worker, noises in zip(self.workers, noises_list)]
            ops = [worker.rollout.remote() for worker in self.workers]
            res_neg = ray.get(ops)
            rollout_neg_vals,r_idx = np.zeros(self.n_workers),0
            for rew,eplen in res_neg:
                rollout_neg_vals[r_idx] = rew
                r_idx = r_idx + 1
                n_env_step += eplen

            b = self.n_workers // 5

            # Scale reward
            rollout_pos_vals, rollout_neg_vals = rollout_pos_vals / 100, rollout_neg_vals / 100

            # Reward
            rollout_concat_vals = np.concatenate((rollout_pos_vals, rollout_neg_vals))
            rollout_delta_vals = rollout_pos_vals - rollout_neg_vals  # pos-neg
            rollout_max_vals = np.maximum(rollout_pos_vals, rollout_neg_vals)
            rollout_max_val = np.max(rollout_max_vals)  # single maximum
            rollout_delta_max_val = np.max(np.abs(rollout_delta_vals))

            # Re-initialize
            rollout_pos_vals, rollout_neg_vals = np.array([]), np.array([])

            # Sort
            sort_idx = np.argsort(-rollout_max_vals)

            # Update
            sigma_R = np.std(rollout_concat_vals)
            weights_updated = []
            for w_idx, weight in enumerate(weights):  # for each weight
                delta_weight_sum = np.zeros_like(weight)
                for k in range(b):
                    idx_k = sort_idx[k]  # sorted index
                    rollout_delta_k = rollout_delta_vals[k]
                    noises_k = noises_list[k]
                    noise_k = (1 / self.nu) * noises_k[w_idx]  # noise for current weight
                    delta_weight_sum += rollout_delta_k * noise_k
                delta_weight = (self.alpha / (b * sigma_R)) * delta_weight_sum
                weight = weight + delta_weight
                weights_updated.append(weight)

            # Set weight
            self.R.set_weights(weights_updated)

            # Print
            if (t == 0) or (((t + 1) % self.print_every) == 0):
                print("[%d/%d] rollout_max_val:[%.2f] rollout_delta_max_val:[%.2f] sigma_R:[%.2f] " %
                      (t, self.total_steps, rollout_max_val, rollout_delta_max_val, sigma_R))

            # Evaluate
            if (t == 0) or (((t + 1) % self.evaluate_every) == 0) or (t == (self.total_steps - 1)):
                ram_percent = psutil.virtual_memory().percent  # memory usage
                print("[Evaluate] step:[%d/%d][%.1f%%] #step:[%.1e] time:[%s] ram:[%.1f%%]." %
                      (t + 1, self.total_steps, t / self.total_steps * 100,
                       n_env_step,
                       time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)),
                       ram_percent)
                      )
                for eval_idx in range(self.num_eval):
                    o, d, ep_ret, ep_len = eval_env.reset(), False, 0, 0
                    if RENDER_ON_EVAL:
                        _ = eval_env.render(mode='human')
                    while not (d or (ep_len == self.max_ep_len_eval)):
                        a = self.R.get_action(o.reshape(1, -1))
                        o, r, d, _ = eval_env.step(a)
                        if RENDER_ON_EVAL:
                            _ = eval_env.render(mode='human')
                        ep_ret += r  # compute return
                        ep_len += 1
                    print(" [Evaluate] [%d/%d] ep_ret:[%.4f] ep_len:[%d]"
                          % (eval_idx, self.num_eval, ep_ret, ep_len))
                latest_100_score.append(ep_ret)
                self.write_summary(t, latest_100_score, ep_ret, n_env_step, time.time() - start_time, rollout_max_val, rollout_delta_max_val, sigma_R)
                print("Saving weights...")
                self.R.save_weight(self.log_path)

        print("Done.")
        eval_env.close()
        ray.shutdown()

    def write_summary(self, episode, latest_100_score, episode_score, total_step, time, rollout_max_val, rollout_delta_max_val, sigma_R):
        with self.summary_writer.as_default():
            tf.summary.scalar("Reward (clipped)", episode_score, step=episode)
            tf.summary.scalar("Latest 100 avg reward (clipped)", np.mean(latest_100_score), step=episode)
            tf.summary.scalar("Total Frames", total_step, step=episode)
            tf.summary.scalar("Time", time, step=episode)
            tf.summary.scalar("rollout_max_val", rollout_max_val, step=episode)
            tf.summary.scalar("rollout_delta_max_val", rollout_delta_max_val, step=episode)
            tf.summary.scalar("sigma_R", sigma_R, step=episode)


    def play(self, load_dir=None, trial=5):
        eval_env = get_eval_env()

        if load_dir:
            loaded_ckpt = tf.train.latest_checkpoint(load_dir)
            self.R.load_weight(loaded_ckpt)

        for i in range(trial):
            o, d, ep_ret, ep_len = eval_env.reset(), False, 0, 0
            if RENDER_ON_EVAL:
                _ = eval_env.render(mode='human')
            while not (d or (ep_len == self.max_ep_len_eval)):
                a = self.R.get_action(o.reshape(1, -1))
                o, r, d, _ = eval_env.step(a)
                if RENDER_ON_EVAL:
                    _ = eval_env.render(mode='human')
                ep_ret += r  # compute return
                ep_len += 1
            print("[Evaluate] [%d/%d] ep_ret:[%.4f] ep_len:[%d]"
                  % (i, trial, ep_ret, ep_len))

def get_env():
    import pybullet_envs,gym
    gym.logger.set_level(40) # gym logger
    return gym.make('AntBulletEnv-v0')

def get_eval_env():
    import pybullet_envs,gym
    gym.logger.set_level(40) # gym logger
    eval_env = gym.make('AntBulletEnv-v0')
    if RENDER_ON_EVAL:
        _ = eval_env.render(mode='human') # enable rendering
    _ = eval_env.reset()
    for _ in range(3): # dummy run for proper rendering
        a = eval_env.action_space.sample()
        o,r,d,_ = eval_env.step(a)
        time.sleep(0.01)
    return eval_env
