import numpy as np
import random
from collections import defaultdict

def epsilon_greedy(Q, state, num_actions, eps):
    if random.random() > eps:
        return np.argmax(Q[state])
    else:
        return random.choice(np.arange(num_actions))

def sarsa_0(env, num_episodes, alpha, gamma=1.0):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    for e in range(num_episodes):
        eps = 1/(e+1)

        state = env.reset()
        action = epsilon_greedy(Q, state, env.action_space.n, eps)

        while True:
            next_state, reward, done, info = env.step(action)
            next_action = epsilon_greedy(Q, next_state, env.action_space.n, eps)

            Q[state][action] = Q[state][action] + alpha*(reward + gamma*Q[next_state][next_action] - Q[state][action])
            state = next_state
            action = next_action
            
            if done:
                break

    return Q

def sarsa_lambda(env, val_lambda, num_episodes, alpha, gamma=1.0):
    Q_new = defaultdict(lambda: np.zeros(env.action_space.n))
    Q_old = defaultdict(lambda: np.zeros(env.action_space.n))

    for e in range(num_episodes):
        eps = 1/(e+1)
        E = defaultdict(lambda: np.zeros(env.action_space.n))

        state = env.reset()
        action = epsilon_greedy(Q_old, state, env.action_space.n, eps)

        while True:
            next_state, reward, done, info = env.step(action)
            next_action = epsilon_greedy(Q_old, next_state, env.action_space.n, eps)
            
            delta = reward + gamma*Q_old[next_state][next_action] - Q_old[state][action]

            E[state][action] += 1
            for s in range(env.observation_space.n):
                for a in range(env.action_space.n):
                    Q_new[s][a] = Q_new[s][a] + alpha*delta*E[s][a]
                    E[s][a] = gamma*val_lambda*E[s][a]

            state = next_state
            action = next_action

            Q_old = Q_new.copy()
            
            if done:
                break

    return Q_new