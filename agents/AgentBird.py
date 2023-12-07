import numpy as np
import math

# Classe base para os agentes
class AgentBird():
    def __init__(
        self,
        prob,
        epsilon,
        alpha,
        gamma,
        epsilonDecay,
        yDiffPrecision,
        yVelPrecision,
        xPosPrecision
    ) -> None:
        self.prob = prob
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.epsilonDecay = epsilonDecay
        # Como o estado é contínuo, é necessário discretizá-lo
        # As variáveis de precisão indicam quantos valores possíveis cada variável pode assumir
        self.yDiffPrecision = yDiffPrecision
        self.yVelPrecision = yVelPrecision
        self.xPosPrecision = xPosPrecision
        
        # Intervalos para os possíveis valores de cada variável
        size = (
            math.floor(1.03 * yDiffPrecision) + 1, # Diferença de altura
            math.floor(2 * yVelPrecision) + 1, # Diferença de posição horizontal
            math.floor(1.05 * xPosPrecision) + 1, # Velocidade vertical
            2 # Ação
        )
        self.q_table = np.zeros(size)
    
    def filter_state(self, state):
        # 0: a posição horizontal do último cano
        # 1: a posição vertical do último cano superior
        # 2: a posição vertical do último cano inferior
        # 3: a posição horizontal do próximo cano
        # 4: a posição vertical do próximo cano superior
        # 5: a posição vertical do próximo cano inferior
        # 6: a posição horizontal do próximo próximo cano
        # 7: a posição vertical do próximo próximo cano superior
        # 8: a posição vertical do próximo próximo cano inferior
        # 9: a posição vertical do jogador
        # 10: a velocidade vertical do jogador
        # 11: a rotação do jogador
        x_last = state[0]
        y_bottom_last = state[2]
        x_pos = state[3]
        y_bottom = state[5]
        y_player = state[9]
        y_vel = state[10]
        
        # Se o "último" estiver à frente de -0.05, então ele é o próximo/atual
        if x_last > -0.05:
            x_pos = x_last + 0.05
            y_bottom = y_bottom_last
        
        # Altura 1 significa que o cano está fora da tela
        if y_bottom == 1:
            # Altura média dos canos
            y_bottom = 0.4892578125
        
        # Ignorando situacoes em que o jogador está fora da tela
        if y_player < 0:
            y_player = 0
        
        y_diff = math.floor((y_bottom - y_player + 0.4) * self.yDiffPrecision)
        y_vel = math.floor((y_vel + 1) * self.yVelPrecision)
        x_pos = math.floor(x_pos * self.xPosPrecision)
        return (y_diff, y_vel, x_pos)
    
    def adjust_reward(state, reward):
        # Ajuste de recompensa para incentivar o jogador a aproximar-se da abertura
        y_top = state[4]
        y_bottom = state[5]
        y_player = state[9]
    
        if y_player < y_top:
            reward -= y_top - y_player
        elif y_player > y_bottom:
            reward -= y_player - y_bottom
        
        return reward
    
    def act(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(np.arange(2), p=[1-self.prob, self.prob])
        
        else:
            value_0, value_1 = self.q_table[state]
            if value_0 < value_1:
                action = 1
            else:
                action = 0
        
        return action
    
    def train(self, env, iter, iterPerEval, evalIter):
        pass
    
    def evaluate(self, env, iter):
        total_reward = 0
        max_reward = 0
        for i in range(iter):
            state = env.reset()
            state = self.filter_state(state[0])
            total = 0
            
            while True:
                action = self.act(state)
                next_state, reward, terminated, _, _ = env.step(action)
                reward = AgentBird.adjust_reward(next_state, reward)
                total += reward
                next_state = self.filter_state(next_state)
                
                state = next_state
                if terminated:
                    break
            
            total_reward += total
            if total > max_reward:
                max_reward = total
        
        print("Average reward:", total_reward / iter)
        print("Max reward:", max_reward)
        
        return total_reward / iter, max_reward

    def play(self, env):
        state = env.reset()
        state = self.filter_state(state[0])
        total = 0
        
        while True:
            action = self.act(state)
            next_state, reward, terminated, _, _ = env.step(action)
            reward = AgentBird.adjust_reward(next_state, reward)
            total += reward
            next_state = self.filter_state(next_state)
            
            state = next_state
            if terminated:
                break
        
        print("Total reward:", total)
    
    def save(self, file):
        np.save(file, self.q_table)
    
    def load(self, file):
        self.q_table = np.load(file)