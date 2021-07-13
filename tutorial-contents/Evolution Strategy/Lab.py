import numpy as np
import matplotlib.pyplot as plt

N_DIMS = 5  # DNA size
DNA_SIZE = 10             # DNA (real number)
DNA_BOUND = [0, 20]       # solution upper and lower bounds
N_GENERATIONS = 200
POP_SIZE = 100           # population size
N_KID = 50               # n kids per generation

TargePos=[10,10]

def MakePnt():
   return np.random.rand(N_DIMS, 2)

def GetFitness(lens):
   arr=[]
   for len in lens:
      arr.append(1/(len-5))
   return arr

# 获取所有样本的长度
def GetLen(xys):
   # 样本所有点到（0,0）的距离
   sum=[]
   for xy in xys:
      xl,yl = xy.reshape((2, 5))
      len=np.sum(np.sqrt((xl-TargePos[0])**2+(yl-TargePos[1])**2))
      sum.append(len)
   return sum
# 计算DNA内最近点的距离
def getMinDisToOther(DNAS):
   sum=[]
   for DNA in DNAS:
      minDis=100000
      xl,yl = DNA.reshape((2, 5))
      for i in range(5):
         for j in range(i + 1,5):
            len=np.sum(np.sqrt((xl-TargePos[0])**2+(yl-TargePos[1])**2))
            minDis=min(minDis,len)
      sum.append(minDis)
   return sum

# 生小孩
def make_kid(pop, n_kid):
    # generate empty kid holder
    kids = {'DNA': np.empty((n_kid, DNA_SIZE))}
    kids['mut_strength'] = np.empty_like(kids['DNA'])
    for kv, ks in zip(kids['DNA'], kids['mut_strength']):
        # crossover (roughly half p1 and half p2)
        # 选父母
        p1, p2 = np.random.choice(np.arange(POP_SIZE), size=2, replace=False)
        # 交叉点
        cp = np.random.randint(0, 2, DNA_SIZE, dtype=np.bool)  # crossover points
        # 分别选择父母的部分DNA
        kv[cp] = pop['DNA'][p1, cp]
        kv[~cp] = pop['DNA'][p2, ~cp]
        # 合并到一个样本中
        ks[cp] = pop['mut_strength'][p1, cp]
        ks[~cp] = pop['mut_strength'][p2, ~cp]

        # 正态分布标准差
        # mutate (change DNA based on normal distribution)
        ks[:] = np.maximum(ks + (np.random.rand(*ks.shape)-0.5), 0.)    # must > 0
        # 正态分布
        kv += ks * np.random.randn(*kv.shape)
        # 限制范围
        kv[:] = np.clip(kv, *DNA_BOUND)    # clip the mutated value
    return kids

# 移除不好样本
def kill_bad(pop, kids):
    # 新老合并
   for key in ['DNA', 'mut_strength']:
      pop[key] = np.vstack((pop[key], kids[key]))

   # 获取所有适应度
   lens=GetLen(pop['DNA'])
   fitness = GetFitness(lens)      # calculate global fitness
   minDis=getMinDisToOther(pop['DNA'])
   fitness+=minDis
   idx = np.arange(pop['DNA'].shape[0])
   # 递增排列，取后POP_SIZE位
   good_idx = idx[np.argsort(fitness)][-POP_SIZE:]   # selected by fitness ranking (not value)
   for key in ['DNA', 'mut_strength']:
      pop[key] = pop[key][good_idx]
   return pop

class SmartDim(object):
   def __init__(self):
      self.pop = dict(DNA=10 * np.random.rand(1, DNA_SIZE).repeat(POP_SIZE, axis=0),   # initialize the pop DNA values
           mut_strength=np.random.rand(POP_SIZE, DNA_SIZE))                # initialize the pop mutation strength values

   def Myplotting(self):
      plt.cla()

      # plt.scatter(self.city_pos[:, 0].T, self.city_pos[:, 1].T, s=100, c='k')
      # plt.scatter(self.pop.)
      plt.xlim((-0.1, 1.1))
      plt.ylim((-0.1, 1.1))
      plt.pause(0.01)

sd =SmartDim()
print(GetLen(sd.pop['DNA']))

for i in range(N_GENERATIONS):


   # print(xl)
   # print(yl)
   # print(GetLen(sd.pop['DNA'][i]))
   plt.pause(0.2)

   kids = make_kid(sd.pop, N_KID)
   xl,yl = sd.pop['DNA'][1].reshape((2, 5))
   sd.pop = kill_bad(sd.pop,kids)
   if 'sca' in globals(): sca.remove()
   sca = plt.scatter(xl, yl, s=200, lw=0, c='red',alpha=0.5);

# print(sd.pop['DNA'])
plt.ioff(); plt.show()