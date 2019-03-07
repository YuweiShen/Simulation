# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:43:10 2019

@author: Yuwei Shen
"""
import numpy as np
import math
import matplotlib.pyplot as plt

# 产生随机数,存储在array中
def runif(n,a=0,b=1):
    '''
    生成n个[a,b]均匀分布的随机数
    input param n: 产生的随机数个数
    input param a: 均匀分布的区间左端点
    input param b: 均匀分布的区间右端点
    return: 一个np.array对象
    
    '''
    unif_array=np.random.rand(n)
    unif_array=np.dot(unif_array,(b-a))
    unif_array+=a
    return unif_array

def rexp(n,beta=1):
    '''
    生成n个服从exp(1\beta)的随机数
    input param n: 产生的随机数个数
    input param beta: 指数分布的参数
    return: 一个np.array对象
    '''
    unif_array=np.random.rand(n)
    log=np.vectorize(math.log)
    unif_array=log(unif_array)
    unif_array=-np.dot(unif_array,beta)
    return unif_array
    
def rnorm(n,mu=0,sigma=1):
    '''
    生成n个服从均值为mu,标准差为sigma的正态分布随机数
    input param n: 随机数个数
    input param mu: 正态分布均值
    input param sigma: 正态分布方差
    return: 一个np.array对象
    '''
    unif1_array,unif2_array=np.random.rand(n),np.random.rand(n)
    log=np.vectorize(math.log)
    sqrt=np.vectorize(math.sqrt)
    sin=np.vectorize(math.sin)
    unif1_array=sqrt(np.dot(-2,log(unif1_array)))
    unif2_array=sin(np.dot(2*math.pi,unif2_array))
    unif_array=np.multiply(unif1_array,unif2_array)
    return unif_array

