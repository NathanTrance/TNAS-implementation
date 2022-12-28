import random
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import TNAS.Train_tester
import time
from nas_201_api import NASBench201API as API

# Operation tree design
node_ops = [[]]*9 # operations on id-th node
e = [[]]*9 # edge list
par = [-1]*9 # parent list

e[0].extend([1,2])
e[2].extend([3,4])
e[3].extend([5,6])
e[4].extend([7,8])

par[1] = 0
par[2] = 0
par[3] = 2
par[4] = 2
par[5] = 3
par[6] = 3
par[7] = 4
par[8] = 4


node_ops[0] = ['none','nor_conv_1x1','nor_conv_3x3','skip_connect','avg_pool_3x3']
node_ops[1] = ['none']
node_ops[2] = ['nor_conv_1x1','nor_conv_3x3','skip_connect','avg_pool_3x3']
node_ops[3] = ['nor_conv_1x1','nor_conv_3x3']
node_ops[4] = ['skip_connect','avg_pool_3x3']
node_ops[5] = ['nor_conv_1x1']
node_ops[6] = ['nor_conv_3x3']
node_ops[7] = ['skip_connect']
node_ops[8] = ['avg_pool_3x3']

# Cell's DAG edge list
L = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]

# Encoding candidates and operations
id_op = ['none','nor_conv_1x1','nor_conv_3x3','skip_connect','avg_pool_3x3']
masks = []

def To_mask(num):
    res = ''
    while(num > 0):
        res += str(num % 2)
        num = int(num/2)
    while(len(res) < 6):
        res += '0'
    res = res[::-1]
    return res

def Build_masks():
    mask_len = 2**6 - 1
    for i in range(mask_len + 1):
        masks.append(To_mask(i))

def Build_cand_arch(Arch, mask, o0, o1):
    Cand_arch = []
    Len = len(mask)
    for i in range(Len):
        b = mask[i]
        op = o0
        if b == '1':
            op = o1
        if Arch[i] == par[op]:
            Cand_arch.append(op)
        else:
            Cand_arch.append(Arch[i])
    return Cand_arch

def Score(Arch):
    return TNAS.Train_tester.get_metric(Arch)

def BFS_T_o():
    Arch = [0,0,0,0,0,0]
    q = []
    q.insert(0,0) # queue.push()
    while(len(q) > 0):
        u = q[-1] # queue.top()
        print(u)
        q.pop() # queue.pop()
        for v in e[u]:
            q.insert(0,v)
        if len(e[u]) == 0:
            continue
        o0 = e[u][0]
        o1 = e[u][1]
        Cur_arch = Build_cand_arch(Arch, To_mask(0), o0, o1)
        score = Score(Cur_arch)
        for mask in masks:
            Cand_arch = Build_cand_arch(Arch, mask, o0, o1)
            print(Cand_arch)
            start_time = time.time()
            cand_score = Score(Cand_arch)
            if score < cand_score:
                Cur_arch = Cand_arch
                score = cand_score
            print(Cand_arch, time.time() - start_time)
        Arch = Cur_arch
    print(Arch)

Build_masks()
BFS_T_o()