import numpy as np
import random
from collections import defaultdict

def epsilon_greedy_policy(Q, num_actions, eps):
    def policy(state):
        if random.random() > eps:
            return np.argmax(Q[state])
        else:
            return random.choice(np.arange(num_actions))
    return policy

def generate_episode(env, policy, max_episode_length=100):
    episode = []
    state = env.reset()
    for t in range(max_episode_length):
        action = policy(state)
        next_state, reward, done, _ = env.step(action)
        episode.append((t, state, action, reward))
        if done:
            break
        state = next_state
    return episode

def first_visit_mc_control(env, num_episodes, gamma=1.0, max_episode_length=100):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: 0.0)

    for i in range(num_episodes):
        eps = 1/(i+1)
        policy = epsilon_greedy_policy(Q, env.action_space.n, eps)
        episode = generate_episode(env, policy, max_episode_length)

        first_visit = defaultdict(int)
        for e in episode:
            if (e[1], e[2]) not in first_visit.keys():
                first_visit[(e[1], e[2])] = e[0]
                        
        G = 0
        for e in episode[::-1]:
            G = gamma*G + e[3]
            if first_visit[(e[1], e[2])] == e[0]:
                N[(e[1], e[2])] += 1
                Q[(e[1], e[2])] = Q[(e[1], e[2])] + (G - Q[(e[1], e[2])])/N[(e[1], e[2])]

    return Q

def every_visit_mc_control(env, num_episodes, gamma=1.0, max_episode_length=100):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: 0.0)

    for i in range(num_episodes):
        eps = 1/(i+1)
        policy = epsilon_greedy_policy(Q, env.action_space.n, eps)
        episode = generate_episode(env, policy, max_episode_length)

        G = 0
        for e in episode[::-1]:
            G = gamma*G + e[3]
            N[(e[1], e[2])] += 1
            Q[(e[1], e[2])] = Q[(e[1], e[2])] + (G - Q[(e[1], e[2])])/N[(e[1], e[2])]

    return Q

