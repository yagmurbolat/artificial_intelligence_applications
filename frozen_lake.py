import gym
import numpy as np 
environment = gym.make('FrozenLake-v1', is_slippery=False)
Q = np.zeros([environment.observation_space.n,environment.action_space.n])
eta = .628
gma = .9
loop = 5000
score_list = []
for i in range(loop):
    s = environment.reset()
    score_tumu = 0
    d = False
    j = 0
    while j < 99:
        environment.render()
        j+=1
        a = np.argmax(Q[s,:] + np.random.randn(1,environment.action_space.n)*(1./(i+1)))
        s1,o,d,_ = environment.step(a)
        Q[s,a] = Q[s,a] + eta*(o + gma*np.max(Q[s1,:]) - Q[s,a])
        score_tumu += o
        s = s1
        if d == True:
            break
    score_list.append(score_tumu)
    environment.render()
print("Elde edilen Q Tablosu:")
print(Q)
print("Ortalama skor değeri:" + str(sum(score_list)/loop)) 