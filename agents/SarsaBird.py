from agents.AgentBird import AgentBird

# Agente SARSA
class SarsaBird(AgentBird):
    def train(self, env, iter, iterPerEval, evalIter):
        reward_info = []
        for i in range(iter):
            if (i + 1) % 5000 == 0:
                print(round(100 * i / iter, 2), "%", sep="")
                
            if self.epsilonDecay:
                self.epsilon *= 0.9999
            
            state = env.reset()
            state = self.filter_state(state[0])
            action = self.act(state)
            
            while True:
                # time.sleep(0.1)
                next_state, reward, terminated, _, _ = env.step(action)
                reward = AgentBird.adjust_reward(next_state, reward)
                next_state = self.filter_state(next_state)
                
                next_action = self.act(next_state)
                next_reward = self.q_table[next_state + (next_action,)]
                
                state_action = state + (action,)
                self.q_table[state_action] += self.alpha * (reward + self.gamma * next_reward - self.q_table[state_action])
                
                state = next_state
                action = next_action
                if terminated:
                    break
            
            if i % iterPerEval == 0:
                reward_info.append(self.evaluate(env, evalIter))
        
        return reward_info