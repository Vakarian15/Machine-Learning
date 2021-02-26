import numpy as np
import time
import argparse
import csv
from collections import defaultdict

class Gridworld(object):
    def __init__(self, world):
        self.width = world.shape[1]
        self.height = world.shape[0]
        self.m = [[-1, 0], [0, 1], [1, 0], [0, -1]]  # up, right, down, left
        self.world=world
        self._start()

    def _start(self):
        self.x = self.height-1
        self.y = 0

    def current(self):
        return tuple((self.x, self.y))

    def clip(self, x, y):
        x = max(x, 0)
        x = min(x, self.height - 1)
        y = max(y, 0)
        y = min(y, self.width - 1)
        return x, y

    def move(self, action):
        self.x += self.m[action][0]
        self.y += self.m[action][1]
        self.x, self.y = self.clip(self.x, self.y)
        return tuple((self.x, self.y))

def epsilon_greedy(Q, state, epsilon=0.4):
    action = np.argmax(Q[state])
    prob = np.ones(4, dtype=np.float32) * epsilon / 4
    prob[action] += 1 - epsilon
    plan_action=np.random.choice(np.arange(4), p=prob)
    return plan_action

def transition(action, p):
    prob = np.zeros(4, dtype=np.float32)
    prob[action] = p
    prob[(action+1)%4] = (1-p)/2
    prob[(action+3)%4] = (1-p)/2
    actual_action=np.random.choice(np.arange(4), p=prob)
    return actual_action

def print_policy(Q, world):
    map = Gridworld(world)
    for i in range(map.height):
        line = ""
        for j in range(map.width):
            action = np.argmax(Q[(i, j)])
            if action == 0:
                line += str((i, j))+"up"+"\t"
            elif action == 1:
                line += str((i, j))+"right"+"\t"
            elif action == 2:
                line += str((i, j))+"down"+"\t"
            else:
                line += str((i, j))+"left"+"\t"
        print(line)

def print_Qvalue(Q, world):
    map = Gridworld(world)
    for i in range(map.height):
        line = ""
        for j in range(map.width):
                line += str((i, j))+str(round(max(Q[(i,j)]),2))+"\t"
        print(line)

def Qlearning(world, epsilon, reward, p, t=20, alpha=0.1, discount_factor=1):
    map = Gridworld(world)
    start_time=time.time()
    n=0
    Q = defaultdict(lambda: np.zeros(4))
    while time.time()-start_time < t: #end, when time>20s
        n+=1
        map._start()
        cur_state= map.current()

        while map.world[cur_state]==0:  #if not terminal state
            plan_action = epsilon_greedy(Q, cur_state, epsilon)  # explore policy
            action=transition(plan_action, p)   #run transition model to get actual action
            next_state= map.move(action)  # apply action to find next state
            if map.world[next_state]!=0:    #if next state is terminal state
                Q[cur_state][action] = Q[cur_state][action] + alpha * (reward+map.world[next_state] + discount_factor * 0 - Q[cur_state][action])   #update Q value
                break
            else:
                next_action = np.argmax(Q[next_state])
                Q[cur_state][action] = Q[cur_state][action] + alpha * (reward + discount_factor * Q[next_state][next_action] - Q[cur_state][action])    #update Q value
                cur_state = next_state
    return Q,n



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', metavar='F', type=str, nargs=1, help='a csv file, the name of the file cannot contain space')
    parser.add_argument('reward', metavar='R', type=str, nargs=1, help='reward')
    parser.add_argument('possibility', metavar='P', type=str, nargs=1, help='correct move')
    args = parser.parse_args()
    
    #Generate map
    with open(args.file[0], 'r',encoding='UTF-8-sig') as f:
        temp=list(csv.reader(f))
        result=np.array(temp)

    c=result.shape
    d=np.nonzero(result)
    state= np.zeros(c)
    state[d]=result[d]

    print(state)
    r=float(args.reward[0])
    p=float(args.possibility[0])

    Q, n = Qlearning(state, 0.4, r, p)
    print("trails:",n)
    print("Policy:")
    print_policy(Q, state)
    print("Highest Q value:")
    print_Qvalue(Q, state)
    print("reward:")
    m=Gridworld(state)
    print(max(Q[(m.height-1,0)]))

