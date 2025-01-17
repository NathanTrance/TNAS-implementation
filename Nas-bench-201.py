import random
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from nas_201_api import NASBench201API as API

node_ops = ['']*9

node_ops[1] = 'none'
node_ops[5] = 'nor_conv_1x1'
node_ops[6] = 'nor_conv_3x3'
node_ops[7] = 'skip_connect'
node_ops[8] = 'avg_pool_3x3'

# L = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]

# [6,7,1,5,1,7]
# [7, 1, 1, 5, 1, 7] 

def Make_bench_string(Arch):
    ops_Arch = []
    for i in range(6):
        ops_Arch.append(str(node_ops[Arch[i]]))
    return '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.format(ops_Arch[0],ops_Arch[1],ops_Arch[3],ops_Arch[2],ops_Arch[4],ops_Arch[5])

Arch = [1, 1, 5, 1, 1, 1]
api = API('NAS-Bench-201-v1_1-096897.pth', verbose=False)
Arch_string = Make_bench_string(Arch)
# # Arch_string = '|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|'

index = api.query_index_by_arch(Arch_string)
api.show(index)

# Best_arch_CIFAR-10 = [6, 6, 7, 6, 6, 5]
# Best_arch_CIFAR-100 = [6,6,7,6,6,6]

# best_arch_index, highest_valid_accuracy = api.find_best(dataset='cifar100', metric_on_set='test', hp='200')

# print('The best architecture on CIFAR-100 test set with 200-epoch training is: [{:}] {:}'.format(best_arch_index, api.arch(best_arch_index)))

# print(highest_valid_accuracy)