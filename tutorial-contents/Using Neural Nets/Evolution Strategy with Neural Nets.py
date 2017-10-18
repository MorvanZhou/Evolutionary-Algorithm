"""
Simple code for Distributed ES proposed by OpenAI.
Based on this paper: Evolution Strategies as a Scalable Alternative to Reinforcement Learning
Details can be found in : https://arxiv.org/abs/1703.03864

Visit more on my tutorial site: https://morvanzhou.github.io/tutorials/
"""
import numpy as np
import gym
import multiprocessing as mp
import time

N_KID = 10                  # half of the training population
N_GENERATION = 5000         # training step
LR = .05                    # learning rate
SIGMA = .05                 # mutation strength or step size
N_CORE = mp.cpu_count()-1
CONFIG = [
    dict(game="CartPole-v0",
         n_feature=4, n_action=2, continuous_a=[False], ep_max_step=700, eval_threshold=500),
    dict(game="MountainCar-v0",
         n_feature=2, n_action=3, continuous_a=[False], ep_max_step=200, eval_threshold=-120),
    dict(game="Pendulum-v0",
         n_feature=3, n_action=1, continuous_a=[True, 2.], ep_max_step=200, eval_threshold=-180)
][2]    # choose your game


def sign(k_id): return -1. if k_id % 2 == 0 else 1.  # mirrored sampling


class SGD(object):                      # optimizer with momentum
    def __init__(self, params, learning_rate, momentum=0.9):
        self.v = np.zeros_like(params).astype(np.float32)
        self.lr, self.momentum = learning_rate, momentum

    def get_gradients(self, gradients):
        self.v = self.momentum * self.v + (1. - self.momentum) * gradients
        return self.lr * self.v


def params_reshape(shapes, params):     # reshape to be a matrix
    p, start = [], 0
    for i, shape in enumerate(shapes):  # flat params to matrix
        n_w, n_b = shape[0] * shape[1], shape[1]
        p = p + [params[start: start + n_w].reshape(shape),
                 params[start + n_w: start + n_w + n_b].reshape((1, shape[1]))]
        start += n_w + n_b
    return p


def get_reward(shapes, params, env, ep_max_step, continuous_a, seed_and_id=None,):
    # perturb parameters using seed
    if seed_and_id is not None:
        seed, k_id = seed_and_id
        np.random.seed(seed)
        params += sign(k_id) * SIGMA * np.random.randn(params.size)
    p = params_reshape(shapes, params)
    # run episode
    s = env.reset()
    ep_r = 0.
    for step in range(ep_max_step):
        a = get_action(p, s, continuous_a)
        s, r, done, _ = env.step(a)
        # mountain car's reward can be tricky
        if env.spec._env_name == 'MountainCar' and s[0] > -0.1: r = 0.
        ep_r += r
        if done: break
    return ep_r


def get_action(params, x, continuous_a):
    x = x[np.newaxis, :]
    x = np.tanh(x.dot(params[0]) + params[1])
    x = np.tanh(x.dot(params[2]) + params[3])
    x = x.dot(params[4]) + params[5]
    if not continuous_a[0]: return np.argmax(x, axis=1)[0]      # for discrete action
    else: return continuous_a[1] * np.tanh(x)[0]                # for continuous action


def build_net():
    def linear(n_in, n_out):  # network linear layer
        w = np.random.randn(n_in * n_out).astype(np.float32) * .1
        b = np.random.randn(n_out).astype(np.float32) * .1
        return (n_in, n_out), np.concatenate((w, b))
    s0, p0 = linear(CONFIG['n_feature'], 30)
    s1, p1 = linear(30, 20)
    s2, p2 = linear(20, CONFIG['n_action'])
    return [s0, s1, s2], np.concatenate((p0, p1, p2))


def train(net_shapes, net_params, optimizer, utility, pool):
    # pass seed instead whole noise matrix to parallel will save your time
    noise_seed = np.random.randint(0, 2 ** 32 - 1, size=N_KID, dtype=np.uint32).repeat(2)    # mirrored sampling

    # distribute training in parallel
    jobs = [pool.apply_async(get_reward, (net_shapes, net_params, env, CONFIG['ep_max_step'], CONFIG['continuous_a'],
                                          [noise_seed[k_id], k_id], )) for k_id in range(N_KID*2)]
    rewards = np.array([j.get() for j in jobs])
    kids_rank = np.argsort(rewards)[::-1]               # rank kid id by reward

    cumulative_update = np.zeros_like(net_params)       # initialize update values
    for ui, k_id in enumerate(kids_rank):
        np.random.seed(noise_seed[k_id])                # reconstruct noise using seed
        cumulative_update += utility[ui] * sign(k_id) * np.random.randn(net_params.size)

    gradients = optimizer.get_gradients(cumulative_update/(2*N_KID*SIGMA))
    return net_params + gradients, rewards


if __name__ == "__main__":
    # utility instead reward for update parameters (rank transformation)
    base = N_KID * 2    # *2 for mirrored sampling
    rank = np.arange(1, base + 1)
    util_ = np.maximum(0, np.log(base / 2 + 1) - np.log(rank))
    utility = util_ / util_.sum() - 1 / base

    # training
    net_shapes, net_params = build_net()
    env = gym.make(CONFIG['game']).unwrapped
    optimizer = SGD(net_params, LR)
    pool = mp.Pool(processes=N_CORE)
    mar = None      # moving average reward
    for g in range(N_GENERATION):
        t0 = time.time()
        net_params, kid_rewards = train(net_shapes, net_params, optimizer, utility, pool)

        # test trained net without noise
        net_r = get_reward(net_shapes, net_params, env, CONFIG['ep_max_step'], CONFIG['continuous_a'], None,)
        mar = net_r if mar is None else 0.9 * mar + 0.1 * net_r       # moving average reward
        print(
            'Gen: ', g,
            '| Net_R: %.1f' % mar,
            '| Kid_avg_R: %.1f' % kid_rewards.mean(),
            '| Gen_T: %.2f' % (time.time() - t0),)
        if mar >= CONFIG['eval_threshold']: break

    # test
    print("\nTESTING....")
    p = params_reshape(net_shapes, net_params)
    while True:
        s = env.reset()
        for _ in range(CONFIG['ep_max_step']):
            env.render()
            a = get_action(p, s, CONFIG['continuous_a'])
            s, _, done, _ = env.step(a)
            if done: break