import numpy as np
import copy

'''
Iterative policy evaluation
Input : 
    environment: env
        action_space
        observation_space
        P[s][a] : probabilty and next state given present state and action
    policy : action probabilities given state
output:
    V(s) for each state s
    Or, the policy
'''
class DPEnv:
    def __init__(self, env):
        self.env = env
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n
        
        P_shape = np.array(env.P).shape
        assert P_shape[0] == self.num_states
        assert P_shape[1] == self.num_actions

    def policy_evaluation(self, policy, gamma=1, theta=1e-8):
        V = np.zeros(self.num_states)
        
        while True:
            delta = 0
            for s in range(self.num_states):
                v = 0
                for a, action_prob in enumerate(policy[s]):
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        v += action_prob * prob * (reward + gamma*V[mext_state])
                delta = max(delta, np.abs(V[s]-v))
                V[s] = v
            if delta < theta:
                break
        return V

    def q_from_v(self, V, gamma=1):
        Q = np.zeros((self.num_states, self.num_actions))

        for s in range(self.num_states):
            for a in range(self.num_actions):
                for prob, next_state, reward, done in self.env.P[s][a]:
                    Q[a][s] += prob * (reward + gamma*V[next_state])
        return Q

    def policy_improvement(self, V, gamma=1):
        policy = np.zeros((self.num_states, self.num_actions))/self.num_actions
        Q = self.q_from_v(V, gamma)

        for s in range(self.num_states):
            policy[s][np.argmax(Q[s])] = 1

        return policy

    def policy_iteration(self, gamma=1, theta=1e-8):
        policy = np.ones((self.num_states, self.num_actions))/self.num_actions

        while True:
            V = self.policy_evaluation(policy, gamma, theta)
            new_policy = self.policy_improvement(V, gamma)

            if (new_policy == policy).all():
                break

            policy = np.copy(new_policy)

        return policy

    def value_iteration(self, gamma=1, epsilon=1e-6):
        V = np.zeros(self.num_state)
        delta = epsilon * (1-gamma)/(2*gamma)
        while True:
            Q = self.q_from_v(V, gamma)
            V_new = np.max(Q, axis=1)

            if np.max(V_new - V) <= delta:
                break

            V = np.copy(V_new)

        policy = self.policy_improvement(V, gamma)
        return policy, V
