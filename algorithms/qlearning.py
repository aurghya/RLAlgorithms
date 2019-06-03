import numpy as np
import random
from collections import defaultdict

def epsilon_greedy(Q, state, nA, eps):
    """Selects epsilon-greedy action for supplied state.
    
    Params
    ======
        Q (dictionary): action-value function
        state (int): current state
        nA (int): number actions in the environment
        eps (float): epsilon
    """
    if random.random() > eps: # select greedy action with probability epsilon
        return np.argmax(Q[state])
    else:                     # otherwise, select an action randomly
        return random.choice(np.arange(nA))

def q_learning(env, num_episodes, alpha, gamma=1.0):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    for e in range(num_episodes):
        state = env.reset()
        eps = 1/(e+1)

        while True:
            action = epsilon_greedy(Q, state, env.action_space.n, eps)
            next_state, reward, done, info = env.step(action)

            delta = reward + gamma*np.max(Q[next_state]) - Q[state][action]
            Q[state][action] = Q[state][action] + alpha*delta

            state = next_state

            if done:
                break

    return Q

def double_q_learning(env, num_episodes, alpha, gamma=1.0):
    Q1 = np.zeros((env.observation_space.n, env.action_space.n))
    Q2 = np.zeros((env.observation_space.n, env.action_space.n))

    for e in range(num_episodes):
        state = env.reset()
        eps = 1/(e+1)

        while True:
            action = epsilon_greedy(Q1+Q2, state, env.action_space.n, eps)
            next_state, reward, done, info = env.step(action)

            delta = 0
            if random.random()>0.5:
                delta = reward + gamma*np.max(Q2[next_state]) - Q[state][action]
                Q1[state][action] = Q1[state][action] + alpha*delta
            else:
                delta = reward + gamma*np.max(Q1[next_state]) - Q[state][action]
                Q2[state][action] = Q2[state][action] + alpha*delta

            state = next_state

            if done:
                break
    return (Q1+Q2)/2