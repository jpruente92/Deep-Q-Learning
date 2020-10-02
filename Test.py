import random
import time
from collections import deque
from math import log2
import numpy as np

from Sum_tree import Sum_tree_queue, Sum_tree


# buffer_size=1000
# my_tree_queue = Sum_tree_queue(buffer_size=buffer_size)
# for i in range(20*buffer_size):
#     my_tree_queue.add_new_value(i,i,i)
#
# print(log2(buffer_size))
# my_tree_queue.sum_tree.print(my_tree_queue.sum_tree.root)

def sample_double_2(values,probs,batch_size):
    return random.choices(population=values,weights=probs,k=batch_size)

def sample_double(values,probs,batch_size):
    draw = np.random.choice(len(values), batch_size,replace=True,p=probs)
    sample=[values[i] for i in draw]
    return sample

def sample_no_double(values,probs,batch_size):
    draw = np.random.choice(len(values), batch_size,replace=False,p=probs)
    sample=[values[i] for i in draw]
    return sample

def sample_tree_double(tree,batch_size):
    return tree.sample_values(True,batch_size)

def sample_tree_no_double(tree,batch_size):
    return tree.sample_values(False,batch_size)

def compute_sample_probs(values,samples):
    cnt=np.zeros(len(values))
    for s in samples:
        for i,v in enumerate(values):
            if(s == v):
                cnt[i]+=1.0
                break
    cnt/=float(len(samples))
    return cnt

size =200000
batch_size=5
nr_loops=100

start_time=time.time()

values=np.array([np.random.random() for x in range(size)])
priorities=np.array([np.random.randint(0,101) for x in range(size)])
print("building list:",(time.time()-start_time))

start_time=time.time()

tree=Sum_tree()
for x in range(size):
    tree.add_new_value(value=values[x],priority=priorities[x],index=x)
print("building tree:",(time.time()-start_time))


total_sum=tree.root.sum
probs=priorities/total_sum



start_time=time.time()
all_samples=[]
for x in range(nr_loops):
    all_samples=[*all_samples,*sample_no_double(values,probs,batch_size)]
print("np no double:",(time.time()-start_time))
# print("\t",compute_sample_probs(values,all_samples))

start_time=time.time()
all_samples=[]
for x in range(nr_loops):
    all_samples=[*all_samples,*sample_tree_no_double(tree,batch_size)]
print("tree no double:",(time.time()-start_time))
# print("\t",compute_sample_probs(values,all_samples))

start_time=time.time()
all_samples=[]
for x in range(nr_loops):
    all_samples=[*all_samples,*sample_double(values,probs,batch_size)]
print("np double:",(time.time()-start_time))
# print("\t",compute_sample_probs(values,all_samples))

start_time=time.time()
all_samples=[]
for x in range(nr_loops):
    all_samples=[*all_samples,*sample_tree_double(tree,batch_size)]
print("tree double:",(time.time()-start_time))
# print("\t",compute_sample_probs(values,all_samples))

start_time=time.time()
all_samples=[]
for x in range(nr_loops):
    all_samples=[*all_samples,*sample_double_2(values,probs,batch_size)]
print("random double:",(time.time()-start_time))
# print("\t",compute_sample_probs(values,all_samples))
