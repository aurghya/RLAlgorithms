import numpy as np
from collections import defaultdict

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

def first_visit_mc_prediction(env, policy, num_episodes, gamma=1.0, max_episode_length=100):
    V = defaultdict(float)
    N = defaultdict(lambda: 0)

    for i in range(num_episodes):
        episode = generate_episode(env, policy, max_episode_length)

        first_visit = defaultdict(int)
        for e in episode:
            if e[1] not in first_visit.keys():
                first_visit[e[1]] = e[0]

        G = 0
        for e in episode[::-1]:
            G = gamma*G + e[3]
            if first_visit[e[1]] == e[0]:
                N[e[1]] += 1
                V[e[1]] = V[e[1]] + (G - V[e[1]])/N[e[1]]

    return V

def every_visit_mc_prediction(env, policy, num_episodes, gamma=1.0, max_episode_length=100):
    V = defaultdict(float)
    N = defaultdict(lambda: 0.0)

    for i in range(num_episodes):
        episode = generate_episode(env, policy, max_episode_length)

        G = 0
        for e in episode[::-1]:
            N[e[1]] += 1
            G = gamma*G + e[3]
            V[e[1]] = V[e[1]] + (G - V[e[1]])/N[e[1]]

    return V
