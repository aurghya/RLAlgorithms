import numpy as np
from collections import defaultdict

def td_0(env, num_episodes, policy, alpha, gamma=1.0):
    V = defaultdict(lambda: 0.0)

    for e in range(num_episodes):
        state = env.reset()

        while True:
            action = policy(state)
            next_state, reward, done, info = env.step(action)
            V[state] = V[state] + alpha*(reward + gamma*V[next_state] - V[state])
            state = next_state

            if done:
                break
    return V

def td_1(env, num_episodes, policy, alpha, gamma=1.0):
    V_new = defaultdict(lambda: 0.0)
    V_old = defaultdict(lambda: 0.0)

    for e in range(num_episodes):
        state = env.reset()
        E = defaultdict(lambda: 0.0)
        
        while True:
            action = policy(state)
            next_state, reward, done, info = env.step(action)
            E[state] = E[state] + 1
            for s in range(env.observation_space.n):
                V_new[s] = V_new[s] + alpha*(reward + gamma*V_old[next_state] - V_old[s])*E[s]
                E[s] = gamma*E[s]
            V_old = V_new.copy()

            if done:
                break

    return V_new

def td_lambda(env, val_lambda, num_episodes, policy, alpha, gamma=1.0):
    V_new = defaultdict(lambda: 0.0)
    V_old = defaultdict(lambda: 0.0)

    for e in range(num_episodes):
        state = env.reset()
        E = defaultdict(lambda: 0.0)
        
        while True:
            action = policy(state)
            next_state, reward, done, info = env.step(action)
            E[state] = E[state] + 1
            for s in range(env.observation_space.n):
                V_new[s] = V_new[s] + alpha*(reward + gamma*V_old[next_state] - V_old[s])*E[s]
                E[s] = val_lambda*gamma*E[s]
            V_old = V_new.copy()

            if done:
                break

    return V_new

