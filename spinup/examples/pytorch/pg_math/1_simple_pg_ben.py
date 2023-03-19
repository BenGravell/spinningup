import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import SGD, Adam, LBFGS
import numpy as np
import gym
from gym.spaces import Discrete, Box


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a multi layer perceptron
    layers = []
    num_hidden_layers = len(sizes) - 2
    if num_hidden_layers < 0:
        raise ValueError('Must specify at least 2 layer sizes (input and output)')
    for j in range(num_hidden_layers + 1):
        act = activation if j < num_hidden_layers else output_activation
        layer1 = nn.Linear(sizes[j], sizes[j + 1])
        layer2 = act()
        layers += [layer1, layer2]
    return nn.Sequential(*layers)


def train(env_name='CartPole-v0', hidden_sizes=None, lr=1e-2, epochs=50, batch_size=5000, optimizer_str='sgd'):
    if hidden_sizes is None:
        hidden_sizes = [32]
    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # make core of policy network
    layer_sizes = [obs_dim] + hidden_sizes + [n_acts]
    logits_net = mlp(sizes=layer_sizes)

    # make function to compute action distribution
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        return get_policy(obs).sample().item()

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp*weights).mean()

    # make optimizer
    if optimizer_str == 'sgd':
        optimizer = SGD(logits_net.parameters(), lr=lr, momentum=0.9, nesterov=True)
    elif optimizer_str == 'adam':
        optimizer = Adam(logits_net.parameters(), lr=lr)
    elif optimizer_str == 'lbfgs':
        optimizer = LBFGS(logits_net.parameters(), line_search_fn='strong_wolfe', history_size=20)
    else:
        raise ValueError('Invalid optimizer string')

    def train_one_epoch():
        # make some empty lists for logging
        batch_obs = []  # for observations
        batch_acts = []  # for actions
        batch_weights = []  # for R(tau) weighting in policy gradient
        batch_rets = []  # for measuring episode returns
        batch_lens = []  # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()  # first obs comes from starting distribution
        ep_rews = []  # list for rewards accrued throughout ep

        # collect experience by acting in the environment with current policy
        while True:
            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret]*ep_len

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        def closure():
            optimizer.zero_grad()
            batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                      act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                      weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                      )
            batch_loss.backward()
            return batch_loss
        if optimizer_str == 'lbfgs':
            batch_loss = optimizer.step(closure)
        else:
            batch_loss = closure()
            optimizer.step()
        return batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' %
              (i+1, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))
    return logits_net


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=5000)
    parser.add_argument('--optimizer_str', type=str, default='sgd')
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    trained_net = train(env_name=args.env_name, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
                        optimizer_str=args.optimizer_str)
