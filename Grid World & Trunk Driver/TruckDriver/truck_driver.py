import numpy as np 
import argparse

#--------------------------------------------
'''
Class Definition
'''
#--------------------------------------------
class WareHouse:
    '''
        List of Attributes: 
            t: the current time of the environment
            packs: a list of the tuples that correspond to the packages stored in the WareHouse. Each tuple has two elements (t,l) the time of creation and the location of the destination
            prob: the probability of a package being created at time t
            prob_M: the maximum probability of a package being created  
            prob_m: the minimum probability of a package being created 
            prob_C: the change in probability from time t to time t+1, depending on the creation of a package  
            r: a uniform random number in the range [0,1) to be used as a reference to create a package
            l: lenght of the road
    '''
    #-------------------------------
    def __init__(self, l):
        self.t = 0 
        self.packs = []
        self.prob = .15 
        self.prob_M = .25 
        self.prob_m = .05
        self.prob_C = 0.02
        self.r = np.random.rand()
        self.l = l
        
    #-----------------------------------
    def update(self):
        '''Update WareHouse class with an addional clock tick'''    
        #Check probability
        if self.r <= self.prob:
            #Create package
            self.packs.append((self.t, np.random.randint(1,self.l+1)))
            #Update probability
            if self.prob + self.prob_C <= self.prob_M:
                self.prob = round(self.prob + self.prob_C,2)
        else:
            #Package was not created, reduce the probability
            if self.prob - self.prob_C >= self.prob_m:
                self.prob = round(self.prob - self.prob_C,2)

        #Update time and random number
        self.t +=1 
        self.r = np.random.rand()
        
    #-----------------------------------
    def __str__(self):
        out = "Warehouse___ \nTime: %s" % (self.t)
        out += "\nPackages: %s" % (len(self.packs))
        out += "\nPackages: %s" % (self.packs)
        out += "\nProbability: %s" % (self.prob)
        out += "\nRandom: %s" % (self.r)
        return out

#-----------------------------------
class Truck:
    '''
        List of Attributes:
            c: maximum number of packages the truck can load
            packs: the loaded packs the Truck has to deliver
    '''
    #-----------------------------------
    def __init__(self, c):
        self.c = c 
        self.packs = []
    
    #-----------------------------------
    def __str__(self):
        out = "Truck___" 
        out += "\nPackages: %s" % (len(self.packs))
        out += "\nPackages: %s" % (self.packs)
        return out
    
#-----------------------------------
class Environment_:
    '''
        List of Attributes: 
            house: a Warehouse Object
            truck: a Truck Object
    '''
    #-------------------------------
    def __init__(self, l, c, s):
        '''
        List of Attributes: 
            house: a Warehouse Object
            truck: a Truck Object
            l: lenght of the road
            c: maximum number of packages the truck can load
            s: penalty required for starting the truck
        '''
        self.house = WareHouse(l) 
        self.truck = Truck(c)
        self.s = s
    
    #-------------------------------            
    def calculate_penalty(self):
        '''Calculates the penalty for not delivering the packages at the current time'''
        time = self.house.t
        reward = 0
        for i in self.house.packs:
            reward += time - i[0]
        for i in self.truck.packs:
            reward += time - i[0]
        return reward

    #-----------------------------------
    def update(self):
        ''' Update the values of the WareHouse and Truck'''
        reward = -self.calculate_penalty()
        self.house.update()
        return reward
    
    #-----------------------------------
    def deliver(self, gamma):
        ''' Load the truck with as many packages as possible 
            Until the clock tick where the driver returns, all the rewards have been calculated
        '''
        #Include the inial cost of delivering the package
        reward = self.s + 0

        if len(self.house.packs) == 0:
            self.update()
            return reward
        
        #Load the packages and sort them
        self.truck.packs = self.house.packs[0:self.truck.c]
        self.house.packs = self.house.packs[self.truck.c:]
        self.truck.packs.sort(key=lambda x:x[1])
        
        #Set the maximum distance
        distance = self.truck.packs[-1][1]

        #Loop for deliveries time steop in the first way
        for i in range(1,distance+1):
            
            #Add the cost of the delivery
            reward -= self.calculate_penalty()*(gamma**i)
            #print(self.house.t, "__", self.calculate_penalty())
            
            while self.truck.packs[0][1] == i and len(self.truck.packs)>1:
                self.truck.packs.pop(0)
                reward += 30*self.house.l*(gamma**i)
            if i == distance and len(self.truck.packs) == 1:
                self.truck.packs.pop(0)
                reward += 30*self.house.l*(gamma**i)
            
            #Similarly, update the environment
            self.update()
        
        #print("___")
        #Loop for the way back
        for i in range(distance):
            #Add the cost of the delivery
            reward -= self.calculate_penalty()*(gamma**(i+distance))
            #print(self.house.t, "__", self.calculate_penalty())
            #Similarly, update the environment
            self.update()
        reward -= self.calculate_penalty()*(gamma**(2*distance))
        
        return reward

    #-----------------------------------
    def __str__(self):
        out = str(self.house) + "\n" + "\n"
        out += str(self.truck) 
        return out

#-----------------------------------
class Deliver_Q:
    '''
        List of Attributes:
            env_: Environment Object 
            Q: Q values for the different states
    '''
    #-----------------------------------
    def __init__(self, l, c, s, r):
        '''
        List of Attributes: 
            l: lenght of the road
            c: maximum number of packages the truck can load
            s: penalty required for starting the truck
            r: Numpy Random State, used to calcute random numbers without interference with the ones ot the WareHouse 
            e: Epsilon Value. Used to do random elections or make a greedy move
            a: Count of the number of actions that were done
        '''
        self.env_ = Environment_(l,c,s) 
        self.Q = {}
        self.r = np.random.RandomState(r)
        self.e = 0.9
        self.a = []
        self.alpha = 0.1 
        self.gamma = 0.9
        
    #-----------------------------------
    def obtain_state(self):
        '''Return a state evaluation of the world'''
        #Calculate capacity
        if self.env_.truck.c > len(self.env_.house.packs):
            capacity = self.env_.truck.c - len(self.env_.house.packs)
        else:
            capacity = 0
        #Calculate distance
        if len(self.env_.house.packs) > 0:
            distance = max(self.env_.house.packs, key=lambda x:x[1])[1]
        else:
            distance = 0
        #Calculate the capacity and distance as relative values bounded in 5 categories
        capacity = ((capacity / self.env_.truck.c) // .19999999999999)*.2
        distance = ((distance / self.env_.house.l) // .19999999999999)*.2
        
        #Return simple tuple
        return (np.round(capacity,1), np.round(distance,1))

    #-----------------------------------
    def action_reward(self, action):
        '''Choose an action and obtain the reward for the next state
            Action = If 0, waits, else, deliver
        '''
        if action == 0:
            reward = self.env_.update()
        else:
            reward = self.env_.deliver(self.gamma)
        return reward
    
    #-----------------------------------
    def update_Q(self):
        '''Carry an action and update the Q Values'''
        #Obtain state
        state = self.obtain_state()
        t0 = self.env_.house.t
        
        #Check if state is in dictionary
        if state in self.Q.keys():
            #If lower that Epsilon, do a random action. Otherwise, choose best action
            if self.r.random() < self.e:
                action = int(self.r.random() > 0.5)
            else:
                action = np.argmax(self.Q[state])
        else:
            #Create the Q values
            self.Q[state] = [0,0]
            #Decide a random action
            action = int(self.r.random() > 0.5)

        #Elaborate action
        reward = self.action_reward(action)
        new_state = self.obtain_state()

        #Update list of actions
        self.a.append((t0,action,reward))

        #Check if new state not in Q Values
        if new_state not in self.Q.keys():
            self.Q[new_state] = [0,0]

        #Update with Bellman equation
        t1 = self.env_.house.t
        self.Q[state][action] = (1-self.alpha)*self.Q[state][action] + (self.alpha)*(reward + (self.gamma**(t1-t0))*max(self.Q[new_state]))

        

        return reward

    #-----------------------------------
    def __str__(self):
        out = "Q____" + "\n" + "Q: " + "\n" 
        for i in self.Q.keys():
            out += str(i) + ": " + str(self.Q[i]) + "\n"
        out += "Actions: " + str(self.a) + "\n" + "\n" + str(self.env_) 
        return out

#-----------------------------------
def parse_command_line():
    """ Parse the command line arguments.
    Returns:
        (Namespace Object): object with attributes 'capacity', 'length', 'penalty' and 'clock_ticks'
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('capacity', type=int, help='Maximum number of packages to be loaded to the truck')
    parser.add_argument('length', type=int, help='Length of the road')
    parser.add_argument('penalty', type=int, help='Penalty of starting the truck without packages')
    parser.add_argument('clock_ticks', type=int, help='Number of clock ticks to train the package')
    
    return parser.parse_args()

#--------------------------------------------
'''
Train the agent
'''
#--------------------------------------------

if __name__ == "__main__":
    
    #Parse arguments and create data
    args = parse_command_line()

    if args.clock_ticks > -1:
        times = args.clock_ticks
    else:
        times = 100**100
    l = args.length
    c = args.capacity
    s = args.penalty

    #Train Randomly
    dq = Deliver_Q(l=l,c=c,s=s, r=int(100*np.random.rand()))
    dq.e = 1 #Always make a random decition to train the agent
    while dq.env_.house.t < times:
        dq.update_Q()
    #Obtain learned policy
    Q = dq.Q

    #Output
    print("\nOptimal Policy \n")
    print("State \t     Action")
    for i in Q.keys():
        print(i,"\t", np.argmax(Q[i]))
    print("\n*If action equals to 0, it indicates the agent should wait. Else, to deliver")
    print("**If a state is not included in the list, the agent will perform a random action\n")