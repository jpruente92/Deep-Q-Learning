from builtins import print

from Sum_tree import Sum_tree, Sum_tree_element
import numpy as np

np.random.seed(0)
tree =Sum_tree(0)
values = [np.random.randint(100) for i in range(1000)]
priorities = [np.random.randint(100) for i in range(1000)]

cnt=0
nodes=[]
for v,p in zip(values,priorities):
    node= Sum_tree_element(v,p,p,None,None,None,cnt)
    tree.add(node)
    nodes.append(node)
    cnt+=1

print("sum before\t",tree.root.sum)
batch,probs=tree.sample_values(False, 100)
batch,probs=tree.sample_values(False, 100)
batch,probs=tree.sample_values(False, 100)
batch,probs=tree.sample_values(False, 100)
batch,probs=tree.sample_values(False, 100)
batch,probs=tree.sample_values(False, 100)
batch,probs=tree.sample_values(False, 100)
batch,probs=tree.sample_values(False, 100)
values_batch=[node.value for node in batch]
print("values",values)
print("values batch",values_batch)
print ("probs\t",probs)
print("sum of probs\t",np.sum(probs))
print("sum after\t",tree.root.sum)
for node,val in zip(nodes,values):
    tree.update_priority(node,val+1)

tree.print(tree.root)
tree.consistency_check(tree.root)



