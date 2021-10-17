#!/usr/bin/env python
# coding: utf-8

# ## 随机工程进度优化
# 
# 问题设置：项目成本$c_{ij} = i+j$是一个常数，日利率是0.6%,项目完成时间为154天，子工程完成的时间服从U(i,j)形式的正态分布。其中决策变量$x_i$是所有以(i,j)表示的子项目所需贷款的到位时间。
# 

# ### 期望费用最小化模型（EVM）
# 
# $$
# \begin{align*}
# min &\;\; E[C(x,\xi)]\\
# s.t.&\;\; E[T(x,\xi)] \le T^0\\
# &\;\; x\ge 0
# \end{align*}
# $$
# 其中
# 
# $$T(x,\xi) = \max_{(k,n+1)}\{T_k(x,\xi)+\xi_{k,n+1}\}\\
# C(x,\xi) = \sum_{(i,j)}c_{ij}(1+r)^{[T(x,\xi)-x_i]}
# $$

# In[3]:

import time 
import numpy as np
import random 
## 生成决策变量样本x,每个xi都是整数
## 为了让生成的样本点有效，根据图结构生成样本
# 保证前面的x_i小，后面的x_i相对大
def sample_one():
#    x  = [None]*18
    x = np.zeros(8)
    x[0] = 1
    x[1],x[2],x[3] = random.randint(4,7),random.randint(4,10),random.randint(4,13)
    x[4] = random.randint(max(x[1],x[2],13),25)
    x[5] = random.randint(max(x[2],13),28)
    x[6] = random.randint(max(x[2],x[3],16),34)
    x[7] = random.randint(max(x[4],x[5],x[6],37),58)
    return x

def sample_two():
    x = np.zeros(8)
    x[0] = 1
    x[1],x[2],x[3] = random.randint(4,7),random.randint(4,10),random.randint(4,13)
    x[4] = random.randint(13,25)
    x[5] = random.randint(13,28)
    x[6] = random.randint(16,34)
    x[7] = random.randint(37,58)
    return x


def is_notvalid(xarr): #判断x,numpy是否满足图结构,如果不满足，返回不满足的样本标号
    ind_notvalid = []
    for ind in range(len(xarr)):
        x = xarr[ind]
        valid = 0
        if min(x[1],x[2],x[3])>x[0] and min(x[5],x[6],x[4])> x[2] and x[6]> x[3]:
            if x[4] > x[1]:
                valid = 1
        if valid ==0:
            ind_notvalid.append(ind)
    
    return (np.zeros(len(ind_notvalid)),np.asarray(ind_notvalid))


def initiate(n,FieldDR=None):
    Chrom = []
    t1 = time.time()
    for i in range(n):
        Chrom.append(sample_one())
    t2 = time.time()
    print(t2-t1)
    return np.asarray(Chrom)

def initiate1(n):
    Chrom = []
    t1 = time.time()
    while len(Chrom)<n:
        x = sample_two()
        if len(is_notvalid([x])[0]) ==0 :
            Chrom.append(sample_one())
    t2 = time.time()
    print(t2-t1)
    return np.asarray(Chrom)
#%%
parents = initiate(1000) #随机生成两个决策向量
prt = initiate1(1000)
#import matplotlib.pyplot as plt
print(sum((parents.T[7]-parents.T[7].mean())**2))
print(sum((prt.T[7]-prt.T[7].mean())**2))


# In[4]:


import math

# 计算任务开始时间   
def calculate_T(x):
    t = np.zeros(8)
    t[0] = 1
    t[1],t[2],t[3] = max(t[0]+random.randint(3,6),x[1]),max(t[0]+random.randint(3,9),x[2]),max(t[0]+random.randint(3,12),x[3])
    t[4] = max(t[1]+random.randint(6,15),t[2]+random.randint(9,15),x[4])
    t[5] = max(t[2]+random.randint(9,18),x[5])
    t[6] = max(t[2]+random.randint(9,21),t[3]+random.randint(12,21),x[6])
    t[7] = max(t[4]+random.randint(15,24),t[5]+random.randint(18,24),t[6]+random.randint(21,24),x[7])
    return t[7]

##测试
calculate_T(parents[0])


# In[5]:


def percentile(array,alpha,lower=False):
    '''
    input param array: 随机数序列
    input param alpha: 百分位点
    input param lower: 下百分位点,默认True
    return percentile：返回百分位点
    '''
    if alpha>1:
        return
    array=np.sort(array)
    n=len(array)
    if lower==False:
        ind=int(round(n*alpha))
    else:
        ind=int(round(n*(1-alpha)))
    return array[ind]


# In[6]:


# 计算花费            
def calculate_C(totaltime,r,x):
    SUM = 0
    SUM += (3+4+5)*(1+r)**math.ceil(totaltime -x[0]) #1开始
    SUM += 7*(1+r)**math.ceil(totaltime -x[1])
    SUM += (8+9+10)*(1+r)**math.ceil(totaltime -x[2])
    SUM += 11*(1+r)**math.ceil(totaltime -x[3])
    SUM += 13*(1+r)**math.ceil(totaltime -x[4])
    SUM += 14*(1+r)**math.ceil(totaltime -x[5])
    SUM += 15*(1+r)**math.ceil(totaltime -x[6])
    return SUM

def expectation(x,r,n):
    etime = []
    ptime = []
    ecost = []
    c0cost = []
    pcost = []
    count = 0
    for solution in x:
        if count %100 == 0:
            print (count)
        times = []
        costs = []
        #print(solution)
        for i in range(n):
            t = calculate_T(solution)
            c = calculate_C(t,r,solution)
            times.append(t)
            costs.append(c)
        count +=1
        etime.append(sum(times)/n)
        ptime.append(sum(np.asarray(times)<60)/n)
        ecost.append(sum(costs)/n)
        c0cost.append(percentile(costs,0.85))
        pcost.append(sum(np.asarray(costs)<900)/n)
    return [np.asarray(etime),np.asarray(ecost),np.asarray(ptime),np.asarray(c0cost),np.asarray(pcost)]

calculate_C(50,0.06,parents[0])


# In[7]:

## 随机抽样
x = initiate(5000)
x


# In[ ]:


y = expectation(x,0.06,3000)
Etime,Ecost = y[0],y[1]


# In[132]:


#训练神经网络,期望时间
from sklearn.neural_network import MLPRegressor as MLP 
timefunc = MLP(activation='relu', learning_rate='adaptive',max_iter = 5000)
timefunc.fit(x,Etime)


# In[133]:

#期望损失
costfunc = MLP(activation='relu', learning_rate='adaptive',max_iter = 5000)
costfunc.fit(x,Ecost)


# In[137]:


# 定义目标和约束函数
def aim(variables,legV):
    '''其中legV是可行性列向量'''
    #y = expectation(variables,0.06,500)
    cost = np.array([abs(costfunc.predict(variables))])
    constraint = np.array([timefunc.predict(variables)])
#    cost = np.array([y[1]])
#    constraint = np.array([y[0]])
    #print(constraint)
    idx1 = np.where(constraint>60)#采用惩罚方法对于超过60天的方法进行惩罚  
    #print(idx1)
    #print(idx1[0])
    idx2 = is_notvalid(variables)
    exIdx = np.unique(np.hstack([idx1,idx2])) # 得到非可行解个体的下标
    exIdx = exIdx.astype(int)
    legV[exIdx] = 0 # 标记非可行解在种群可行性列向量中对应的值为0(0表示非可行解，1表示可行解)
    return [cost.T,legV]

def punishing(LegV, FitnV):
    FitnV[np.where(LegV == 0)[0]] = 0 # 惩罚非可行解个体的适应度
    return FitnV

aim(parents,np.ones((2,8)))
expectation(np.asarray([0,1,1,2,5,2,3,3]).reshape((1,8)),0.06,1000)

# In[ ]:

## GA算法
import time 
import numpy as np
import sys
import random
import geatpy as ga
## 交叉
rd = np.vectorize(round)
def crossover(parents,recopt):
    POP_SIZE = parents.shape[0]
    sub_pop = [] #子代
    for parent in parents:       
        if np.random.rand() < recopt: #选定交叉的染色体
            i_ = np.random.randint(0, POP_SIZE, size=1) #选另外一个染色体                           
            lamd = random.random() #随机产生另一个数
            subpop1 = rd(lamd*parent+(1-lamd)*parents[i_]).reshape(8) # 小孩1
            subpop2 = rd(lamd*parents[i_]+(1-lamd)*parent).reshape(8) #小孩2 
            sub_pop.append(subpop1)
            sub_pop.append(subpop2)
        else:
            sub_pop.append(parent)
    return np.asarray(sub_pop)

## 评价适应度函数
def judge(ObjV,maxormin,alpha = 0.7):
    #排序
    POP_SIZE = ObjV.shape[0]
    FitnV = np.zeros(POP_SIZE)
    t = list(ObjV.reshape(POP_SIZE))#从小到大
    if maxormin == -1:
        t = sorted(t) #越小越好
    else:
        t = sorted(t,reverse=True) #越大越好
    for i in range(len(t)):
        for j in range(POP_SIZE):
            if t[i] == ObjV[j][0]: #找到对应的数
                FitnV[j] = alpha*(1-alpha)**(i)
                break
    FitnV = FitnV.reshape((POP_SIZE,1))
    return FitnV
                
  
## 变异
def mutation(offspring_crossover,pm):
    mut_pop = []
    for idx in range(offspring_crossover.shape[0]):
        mut_pop.append(offspring_crossover[idx])
        random_value = random.randint(-2,2)
#        loc = np.random.randint(0,offspring_crossover.shape[1],size = 1)
        if np.random.rand()< pm:
#            offspring_crossover[idx][loc] = abs(offspring_crossover[idx][loc] + random_value)
            offspring_crossover[idx] = abs(offspring_crossover[idx] + random_value)
            mut_pop.append(offspring_crossover[idx])
    return np.asarray(mut_pop)

## 自然选择,采用精英策略 + 轮盘赌
def select(Chrom,FitnV,NIND,maxormin):  
    Chrom = Chrom[np.where(FitnV[:,0]!=0)]
    FitnV = FitnV[np.where(FitnV[:,0]!=0)]
    #print(Chrom)
  
    idx = np.random.choice(np.arange(Chrom.shape[0]), size= math.floor(NIND*0.9), replace=True,
                           p=FitnV[:,0]/FitnV.sum()) #先不改变Chrom和适应度的大小关系
    a = Chrom[idx]
    Chrom = Chrom[np.argsort(FitnV[:,0])] #适应度从小到大排列
    #print(idx)
    #print(Chrom[idx])
    return np.vstack([a,Chrom[math.floor(NIND*0.9):]])
   

def GATemplate(AIM, PUN, FieldDR, maxormin, MAXGEN, NIND,recopt, pm,alpha):
    aimfuc = AIM
    if PUN is not None:
        punishing = PUN # 获得罚函数
    if FieldDR is not None:
        NVAR = FieldDR.shape[0] # 得到控制变量的个数
    NVAR = 8
    # 定义进化记录器，初始值为nan
    pop_trace = (np.zeros((MAXGEN ,2)) * np.nan)
    # 定义变量记录器，记录控制变量值，初始值为nan
    var_trace = (np.zeros((MAXGEN ,NVAR)) * np.nan) 
    #print(var_trace)
    # 生成初始种群
    Chrom = initiate(NIND, FieldDR)
    LegV = np.ones((Chrom.shape[0], 1)) # 生成可行性列向量，元素为1表示对应个体是可行解，0表示非可行解
        #print(LegV)
    [ObjV, LegV] = aimfuc(Chrom, LegV) # 求种群的目标函数
    while sum(LegV)==0:
        Chrom = initiate(NIND, FieldDR)
        LegV = np.ones((Chrom.shape[0], 1)) # 生成可行性列向量，元素为1表示对应个体是可行解，0表示非可行解
        #print(LegV)
        [ObjV, LegV] = aimfuc(Chrom, LegV) # 求种群的目标函数
    #print(Chrom)
    gen = 0
    # 开始进化！！
    start_time = time.time() # 开始计时
    badcount = 0
    while gen < MAXGEN:
        #print(gen)
        # 进行遗传算子，生成子代
        SelCh = crossover1(Chrom, recopt) # 重组
        Chrom = mutation1(SelCh, pm) # 变异
        #print(Chrom)
        LegV = np.ones((Chrom.shape[0], 1)) # 生成可行性列向量，元素为1表示对应个体是可行解，0表示非可行解
        #print(LegV)
        [ObjV, LegV] = aimfuc(Chrom, LegV) # 求种群的目标函数
        #print('ObjV',ObjV.shape)
        #print('LegV',LegV.shape)
        FitnV = judge(ObjV,maxormin,alpha)
        if PUN is not None:
            FitnV = punishing(LegV, FitnV) # 把不合格的适应度改成0
        # 记录进化过程
        bestIdx = np.nanargmax(FitnV) # 获取最优个体的下标
        #print(bestIdx)
        if LegV[bestIdx] != 0:#记录可行解
            pop_trace[gen,0] = ObjV[bestIdx] # 记录当代目标函数的最优值
            var_trace[gen,:] = Chrom[bestIdx,:] # 记录当代最优的控制变量值
        else:
            gen -= 1 # 忽略这一代
            badcount += 1 
        if badcount >100:
            break
        if sum(FitnV)!=0:
            #print(FitnV)
            Chrom = select(Chrom, FitnV,NIND,maxormin)
        gen += 1
        if gen % 50 ==0:
            print(gen)
    end_time = time.time() # 结束计时
    times = end_time - start_time
    # 输出结果
    ga.trcplot(pop_trace, [['种群最优个体目标函数值']])
    if maxormin ==1 :
        best_gen = np.nanargmax(pop_trace[:, 0]) # 记录最优种群是在哪一代
        best_ObjV = np.nanmax(pop_trace[:, 0])
    else:
        best_gen = np.nanargmin(pop_trace[:, 0]) # 记录最优种群是在哪一代
        best_ObjV = np.nanmin(pop_trace[:, 0])
    if np.isnan(best_ObjV):
        raise RuntimeError('error: no feasible solution. (没找到可行解。)')
    print('最优的目标函数值为：', best_ObjV)
    print('最优的控制变量值为：')
    for i in range(NVAR):
        print(var_trace[best_gen, i])
    print('最优的一代是第', best_gen + 1, '代')
    print('时间已过', times, '秒')
    # 返回进化记录器、变量记录器以及执行时间
    return [pop_trace, var_trace, times]


# In[1]:
def crossover1(parents,recopt):
    POP_SIZE = parents.shape[0]
    sub_pop = [] #子代
    for parent in parents:
        sub_pop.append(parent)
        if np.random.rand() < recopt: #选定交叉的染色体
            i_ = np.random.randint(0, POP_SIZE, size=1) #选另外一个染色体                           
            cross_points = np.random.randint(0,parents.shape[1] , size=1)   # 选择交叉点，进行平坦交叉
            rd = np.vectorize(round)
            parent[cross_points] = rd((parent[cross_points] + parents[i_, cross_points] )/2) # 小孩
            sub_pop.append(parent)
    return np.asarray(sub_pop)

## 变异
def mutation1(offspring_crossover,pm):
    mut_pop = []
    for idx in range(offspring_crossover.shape[0]):
        mut_pop.append(offspring_crossover[idx])
        random_value = random.randint(-2,2)
        loc = np.random.randint(0,offspring_crossover.shape[1],size = 1)
        if np.random.rand()< pm:
            offspring_crossover[idx][loc] = abs(offspring_crossover[idx][loc] + random_value)
            mut_pop.append(offspring_crossover[idx])
    return np.asarray(mut_pop)



#%%
#def GAtemplate(AIM, PUN, FieldDR, problem, maxormin, GGAP, MAXGEN, NIND, SUBPOP, selectStyle, recombinStyle, recopt, pm, drawing = 1):
#    GGAP = 0.5 # 因为父子合并后选择，因此要将代沟设为0.5以维持种群规模
#    aimfuc = AIM
#    if PUN is not None:
#        punishing = PUN # 获得罚函数
#    NVAR = FieldDR.shape[1] # 得到控制变量的个数
#    # 定义进化记录器，初始值为nan
#    pop_trace = (np.zeros((MAXGEN ,2)) * np.nan)
#    # 定义变量记录器，记录控制变量值，初始值为nan
#    var_trace = (np.zeros((MAXGEN ,NVAR)) * np.nan) 
#    repnum = 0 # 初始化重复个体数为0
#    ax = None # 存储上一帧图形
#    if problem == 'R':
#        Chrom = ga.crtrp(NIND, FieldDR) # 生成初始种群
#    elif problem == 'I':
#        Chrom = initiate(NIND, FieldDR)
#    LegV = np.ones((NIND, 1)) # 生成可行性列向量，元素为1表示对应个体是可行解，0表示非可行解
#    [ObjV, LegV] = aimfuc(Chrom, LegV) # 求种群的目标函数
#    while sum(LegV) == 0:
#        #print(LegV)
#        Chrom = initiate(NIND, FieldDR)
#        LegV = np.ones((Chrom.shape[0], 1)) # 生成可行性列向量，元素为1表示对应个体是可行解，0表示非可行解
#        #print(LegV)
#        [ObjV, LegV] = aimfuc(Chrom, LegV) # 求种群的目标函数
#    gen = 0
#    badCounter = 0 # 用于记录在“遗忘策略下”被忽略的代数
#    # 开始进化！！
#    start_time = time.time() # 开始计时
#    while gen < MAXGEN:
#        if badCounter >= 10 * MAXGEN: # 若多花了10倍的迭代次数仍没有可行解出现，则跳出
#            break
#        # 进行遗传算子，生成子代
#        SelCh=ga.recombin(recombinStyle, Chrom, recopt, SUBPOP) # 重组
#        if problem == 'R':
#            SelCh=ga.mutbga(SelCh,FieldDR, pm) # 变异
#        elif problem == 'I':
#            #SelCh=ga.mutint(SelCh, FieldDR, pm) #变异：整数
#            SelCh = mutation(SelCh,pm)
#        LegVSel = np.ones((SelCh.shape[0], 1)) # 初始化育种种群的可行性列向量
#        [ObjVSel, LegVSel] = aimfuc(SelCh, LegVSel) # 求育种种群的目标函数值
#        # 父子合并
#        Chrom = np.vstack([Chrom, SelCh])
#        ObjV = np.vstack([ObjV, ObjVSel])
#        LegV = np.vstack([LegV, LegVSel])
#        # 对合并的种群进行适应度评价
#        FitnV = ga.ranking(maxormin * ObjV, LegV, None, SUBPOP)
#        if PUN is not None:
#            FitnV = punishing(LegV, FitnV) # 调用罚函数
#        # 记录进化过程
#        bestIdx = np.nanargmax(FitnV) # 获取最优个体的下标
#        if LegV[bestIdx] != 0:
#            feasible = np.where(LegV != 0)[0] # 排除非可行解
#            pop_trace[gen,0] = np.sum(ObjV[feasible]) / ObjV[feasible].shape[0] # 记录种群个体平均目标函数值
#            pop_trace[gen,1] = ObjV[bestIdx] # 记录当代目标函数的最优值
#            var_trace[gen,:] = Chrom[bestIdx, :] # 记录当代最优的控制变量值
#            repnum = len(np.where(ObjV[bestIdx] == ObjV)[0]) # 计算最优个体重复数
#            # 绘制动态图
#            if drawing == 2:
#                ax = ga.sgaplot(pop_trace[:,[1]],'种群最优个体目标函数值', False, ax, gen)
#        else:
#            gen -= 1 # 忽略这一代
#            badCounter += 1
#        [Chrom,ObjV,LegV]=ga.selecting(selectStyle, Chrom, FitnV, GGAP, SUBPOP, ObjV, LegV) # 选择个体生成新一代种群
#        gen += 1
#        if gen % 50 ==0 :
#            print(gen)
#    end_time = time.time() # 结束计时
#    times = end_time - start_time
#    # 绘图
#    if drawing != 0:
#        ga.trcplot(pop_trace, [['种群个体平均目标函数值', '种群最优个体目标函数值']])
#    # 输出结果
#    if maxormin == 1:
#        best_gen = np.nanargmin(pop_trace[:, 1]) # 记录最优种群是在哪一代
#        best_ObjV = np.nanmin(pop_trace[:, 1])
#    if maxormin == -1:
#        best_gen = np.nanargmax(pop_trace[:, 1]) # 记录最优种群是在哪一代
#        best_ObjV = np.nanmax(pop_trace[:, 1])
#    #print(pop_trace)
#    #print(best_ObjV)
#    if np.isnan(best_ObjV):
#        raise RuntimeError('error: no feasible solution. (没找到可行解。)')
#    print('最优的目标函数值为：', best_ObjV)
#    print('最优的控制变量值为：')
#    for i in range(NVAR):
#        print(var_trace[best_gen, i])
#    print('最优的一代是第', best_gen + 1, '代')
#    print('时间已过', times, '秒')
#    # 返回进化记录器、变量记录器以及执行时间
#    return [pop_trace, var_trace, times]

#%%
# 变量设置
variable = [[1,1],]
boundary = [[1,1],]
for i in range(7):
    variable.append([1,60])
    boundary.append([1,1])    
ranges = np.vstack(variable).T       # 生成自变量的范围矩阵
borders = np.vstack(boundary).T      # 生成自变量的边界矩阵
FieldDR = ga.crtfld(ranges, borders) # 生成区域描述器

#[pop_trace, var_trace, times] = GAtemplate(aim, punishing, FieldDR, problem = 'I', maxormin = 1, MAXGEN = 2000, NIND = 100, SUBPOP = 1, GGAP = 0.9, selectStyle = 'rws', recombinStyle = 'xovdp', recopt = 0.9, pm = 0.5, drawing = 1)

[pop_trace, var_trace, times] = GATemplate(aim, punishing, FieldDR=None, maxormin = -1, MAXGEN = 2000, NIND = 30, recopt = 0.3, pm = 0.05,alpha = 0.8)


# ### $\alpha$费用最小模型
# $$
# \begin{align*}
# min &\;\; C^0\\
# s.t.&\;\; Pr\{C(x,\xi)\le C^0\}\ge \alpha\\
# &\;\; Pr\{T(x,\xi)\le T^0\}\ge \beta\\
# &\;\; x\ge 0
# \end{align*}
# $$
# 
# 在这里，分别让$\alpha,\beta$为0.95

# In[170]:


Ptime, C0cost = y[2],y[3]

from sklearn.neural_network import MLPRegressor as MLP 
timecons = MLP(activation='relu', learning_rate='adaptive',max_iter = 5000)
timecons.fit(x,Ptime)
c0cost = MLP(activation='relu', learning_rate='adaptive',max_iter = 5000)
c0cost.fit(x,C0cost)


# In[173]:


## 定义目标函数
def aim(variables,legV):
    '''其中legV是可行性列向量'''
    cost = np.array([c0cost.predict(variables)])
    constraint = np.array([timecons.predict(variables)])
    idx1 = np.where(constraint<0.95)#采用惩罚方法对于概率小于0.9的方法进行惩罚   
    idx2 = is_notvalid(variables)
    exIdx = np.unique(np.hstack([idx1,idx2])) # 得到非可行解个体的下标
    exIdx = exIdx.astype(int)
    legV[exIdx] = 0 # 标记非可行解在种群可行性列向量中对应的值为0(0表示非可行解，1表示可行解)
    return [cost.T,legV]


# In[181]:


#[pop_trace, var_trace, times] = GAtemplate(aim, punishing, FieldDR, problem = 'I', maxormin = 1, MAXGEN = 5000, NIND = 100, SUBPOP = 1, GGAP = 0.9, selectStyle = 'rws', recombinStyle = 'xovdp', recopt = 0.5, pm = 0.3, drawing = 1)

[pop_trace, var_trace, times] = GATemplate(aim, punishing, FieldDR=None, maxormin = 1, MAXGEN = 1000, NIND = 100, recopt = 0.3, pm = 0.05,alpha = 0.7)


#### 期望最小模型2




# In[183]:


def aim(variables,legV):
    '''其中legV是可行性列向量'''
    cost = np.array([costfunc.predict(variables)])
    constraint = np.array([timecons.predict(variables)])
    idx1 = np.where(constraint<0.95)#采用惩罚方法对于超过60的方法进行惩罚   
    idx2 = is_notvalid(variables)
    exIdx = np.unique(np.hstack([idx1,idx2])) # 得到非可行解个体的下标
    exIdx = exIdx.astype(int)
    legV[exIdx] = 0 # 标记非可行解在种群可行性列向量中对应的值为0(0表示非可行解，1表示可行解)
    return [cost.T,legV]


# In[186]:


#[pop_trace, var_trace, times] = GAtemplate(aim, punishing, FieldDR, problem = 'I', maxormin = 1, MAXGEN = 5000, NIND = 100, SUBPOP = 1, GGAP = 0.9, selectStyle = 'rws', recombinStyle = 'xovdp', recopt = 0.5, pm = 0.3, drawing = 1)

[pop_trace, var_trace, times] = GATemplate(aim, punishing, FieldDR=None, maxormin = -1, MAXGEN = 1000, NIND = 30, recopt = 0.3, pm = 0.05,alpha = 0.7)


# In[ ]:


# ## 用不确定理论计算
# $
# \begin{align*}
# \min_x & \int_0^1 \gamma^{-1}(x,\alpha)d\alpha\\
# s.t.
# &\;\;\Psi^{-1}(x,\alpha_0)\le T_0\\
# &\;\; x\ge 0
# \end{align*}
# $
# 
# 其中,$\Psi^{-1}(x,\alpha_0)$是$T(x,\xi)$的逆不确定分布，$\gamma^{-1}(x,\alpha)$是$C(x,\xi)$的逆不确定分布。在这里，让$\alpha_0 = 0.95$
# 
# 可以简化成
# 
# $
# \begin{align*}
# \min_x & (c_1+c_2+\cdots +c_{99})/99\\
# s.t.
# &\;\;k/100 \ge \alpha \text{ if } s_k \ge T^0 \\
# &\;\; x\ge 0
# \end{align*}
# $

# In[214]:


def expect_c(samples):
    timelist = []
    costlist = []
    for sample in samples:
        sample = np.asarray(sample)
        #print(sample)
        t_table = []
        c_table = []
        for i in range(99):
            t = calculate_T(sample)
            #print(t)
            c = calculate_C(t,0.06,sample)
            t_table.append(t)
            c_table.append(c)
        t_table = np.sort(np.asarray(t_table))
        flag = 0
        for ind in range(99):
            if t_table[ind] >= 60:
                flag = 1
                timelist.append(ind+1) #记录首次大于T0的k
                break
        if flag == 0:
            timelist.append(100) #填空
        costlist.append(sum(c_table)/99)
    return [costlist,timelist]
        


# In[217]:


def aim(variables,legV):
    '''其中legV是可行性列向量'''
    a = expect_c(variables)
    cost = np.array([a[0]])
    constraint = np.array([a[1]])
    #print(constraint)
    idx1 = np.where(constraint<95)  #大于60的必须在95以上  
    idx2 = is_notvalid(variables)
    exIdx = np.unique(np.hstack([idx1,idx2])) # 得到非可行解个体的下标
    exIdx = exIdx.astype(int)
    legV[exIdx] = 0 # 标记非可行解在种群可行性列向量中对应的值为0(0表示非可行解，1表示可行解)
    return [cost.T,legV]

#aim(parents,np.ones((2,19)))


# In[224]:


#[pop_trace, var_trace, times] = GATemplate(aim, punishing, FieldDR, problem = 'I', maxormin = 1, MAXGEN = 1000, NIND = 80, SUBPOP = 1, GGAP = 0.9, selectStyle = 'rws', recombinStyle = 'xovdp', recopt = 0.4, pm = 0.3, drawing = 1)

[pop_trace, var_trace, times] = GATemplate(aim, punishing, FieldDR=None, maxormin = -1, MAXGEN = 1000, NIND = 30, recopt = 0.3, pm = 0.05,alpha = 0.7)


# In[ ]:





# In[ ]:




