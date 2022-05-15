import argparse
from ast import AsyncFunctionDef
import numpy as np
from environment import MountainCar, GridWorld
import random
import matplotlib.pyplot as plt


# NOTE: We highly recommend you to write functions for...
# - converting the state to a numpy array given a sparse representation
# - determining which state to visit next

def sparse2np(sparse, mode, state_space):
    if mode == 'raw':
        return np.array([sparse[0], sparse[1]])
    a = np.zeros(state_space)
    for i in sparse:
        a[i] = 1
    return a

def next_state(state, weights, action_space, epsilon):
    if (random.random() < epsilon):
        return random.randint(0, action_space-1)
    x = np.matmul(np.transpose(state), weights)
    return np.argmax(x.flatten())

def main(args):
    # Command line inputs
    mode = args.mode
    weight_out = args.weight_out
    returns_out = args.returns_out
    episodes = args.episodes
    max_iterations = args.max_iterations
    epsilon = args.epsilon
    gamma = args.gamma
    learning_rate = args.learning_rate
    debug = args.debug

    # We will initialize the environment for you:
    if args.environment == 'mc':
        env = MountainCar(mode=mode, debug=debug)
    else:
        env = GridWorld(mode=mode, debug=debug)

    weights = np.zeros((env.state_space, env.action_space))
    bias = 0

    returns = []  # This is where you will save the return after each episode
    for episode in range(episodes):
        # Reset the environment at the start of each episode
        state = sparse2np(env.reset(), mode, env.state_space)  # `state` now is the initial state
        r = 0
        for it in range(max_iterations):
            action = next_state(state, weights, env.action_space, epsilon)
            stat, reward, done = env.step(action)
            state2 = sparse2np(stat, mode, env.state_space)
            q = np.matmul(np.transpose(state), weights[:, action]) + bias
            qq = np.matmul(np.transpose(state2), weights) + bias
            weights[:, action] -= learning_rate * (q - reward - gamma * np.max(qq)) * state
            bias -= learning_rate * (q - reward - gamma * np.max(qq))
            state = state2
            r += reward
            if (done or it == max_iterations-1):
                returns.append(r)
                break

    running = []
    for i in range(episodes - 25):
        val = np.sum(returns[i:i+25])/25
        running.append(val)
    
    x = np.linspace(0, episodes, episodes)
    plt.title("Episodes vs. Returns (tile)")
    plt.xlabel("Episode")
    plt.ylabel("Returns")
    plt.plot(x, returns, "r", linewidth=3.0, label="returns (individual)")
    plt.plot(x[25:], running, "b", linewidth=3.0, label="returns (rolling mean)")
    plt.legend(loc = "upper left")
    plt.show()

    with open(weight_out, 'w') as fout:
        fout.write(str(bias) + "\n")
        for i in weights:
            for j in i:
                fout.write(str(j) + "\n")
    with open(returns_out, 'w') as fout:
        for i in returns:
            fout.write(str(i) + "\n")


if __name__ == "__main__":
    # No need to change anything here
    parser = argparse.ArgumentParser()
    parser.add_argument('environment', type=str, choices=['mc', 'gw'],
                        help='the environment to use')
    parser.add_argument('mode', type=str, choices=['raw', 'tile'],
                        help='mode to run the environment in')
    parser.add_argument('weight_out', type=str,
                        help='path to output the weights of the linear model')
    parser.add_argument('returns_out', type=str,
                        help='path to output the returns of the agent')
    parser.add_argument('episodes', type=int,
                        help='the number of episodes to train the agent for')
    parser.add_argument('max_iterations', type=int,
                        help='the maximum of the length of an episode')
    parser.add_argument('epsilon', type=float,
                        help='the value of epsilon for epsilon-greedy')
    parser.add_argument('gamma', type=float,
                        help='the discount factor gamma')
    parser.add_argument('learning_rate', type=float,
                        help='the learning rate alpha')
    parser.add_argument('--debug', type=bool, default=False,
                        help='set to True to show logging')
    main(parser.parse_args())
