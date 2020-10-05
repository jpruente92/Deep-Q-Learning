import random


from collections import deque

# in the sum tree the elements are saved
# in the queue the node are saved so that they can be removed when max size is reached
class Sum_tree_queue():
    def __init__(self,buffer_size):
        self.sum_tree= Sum_tree()
        self.queue=deque(maxlen=buffer_size)
        self.max_size=buffer_size

    def add_new_value(self,value,priority,index):
        if(self.max_size == len(self.queue)):
            node_to_delete=self.queue[0]
            self.sum_tree.remove(node_to_delete)
        new_node=Sum_tree_element(value,priority,priority,None,None,None,index)
        self.queue.append(new_node)
        self.sum_tree.add(new_node)

    def __len__(self):
        return len(self.queue)



class Sum_tree():
    def __init__(self):
        self.root=None

    # works for trees and sub trees
    def add(self,new_node):
        if self.root is None:
            self.root=new_node
            return
        # always add nodes or sub trees as leaf
        # choose always side with shorter path to leave
        father = self.root
        while True:
            if(father.left is None):
                father.left=new_node
                break
            if(father.right is None):
                father.right=new_node
                break
            if(father.left.min_path_to_leaf<=father.right.min_path_to_leaf):
                father=father.left
            else:
                father=father.right
        new_node.father=father
        # update sum and update_path_lengths of all ancestors
        while True:
            father.sum+=new_node.sum
            father.update_path_lengths()
            if(father.father is None):
                break
            father=father.father

    def add_new_value(self,value,priority,index):
        self.add(Sum_tree_element(value,priority,priority,None,None,None,index))


    # idea: remove the node and add all children to the tree via add
    # if node is root and has two children add one child to the other
    def remove(self,node):
        if self.root is node:
            if(node.left is None):
                if (node.right is None):
                    return
                else:
                    self.root=node.right
            else:
                if (node.right is None):
                    self.root=node.right
                else:
                    # tree with higher max path lengthwill be the root
                    if(node.right.max_path_to_leaf>node.left.max_path_to_leaf):
                        self.root=node.right
                        self.add(node.left)
                    else:
                        self.root=node.left
                        self.add(node.right)
        else:
            father = node.father
            # delete node from father
            if father.left is node:
                father.left=None
            else:
                father.right=None
            # subtract sum from all ancestors of node and update path lengths
            while True:
                father.sum-=node.sum
                father.update_path_lengths()
                if(father.father is None):
                    break
                father=father.father
            # add children to tree
            if(node.left is not None):
                self.add(node.left)
            if(node.right is not None):
                self.add(node.right)

    def update_priority(self,node,priority):
        diff = node.priority-priority
        node.priority=priority
        while node is not None:
            node.sum-=diff
            node=node.father


    def sample_values(self,replace,batch_size):
        random_number = random.random()
        node =self.root
        total_sum=self.root.sum
        nodes=[]
        probabilities=[]
        indices=[]
        for x in range(batch_size):
            while True:
                sum_left=0
                if(node.left is not None):
                    sum_left=node.left.sum
                # if we replace or node s not drawn yet, we have to regard nothing
                if(replace or not node.index in indices):
                    if(node.left is not None and random_number<=sum_left/total_sum):
                        node=node.left
                    elif(node.right is not None and random_number>=(sum_left+node.priority)/total_sum):
                        node=node.right
                        random_number-=((sum_left+node.priority)/total_sum)
                    else:
                        nodes.append(node)
                        probabilities.append(node.priority/total_sum)
                        break
                # otherwise the node has to be ignored
                else:
                    if(node.left is not None and random_number<=sum_left/total_sum):
                        node=node.left
                    else:
                        node=node.right
                        random_number-=((sum_left)/total_sum)
                        # change probabilities according to the removal
                        total_sum-=node.priority
                        indices.append(node.index)
        return nodes,probabilities


    def print(self,node):
        string_left="-"
        if(node.left is not None):
            string_left=node.left.index
        string_right="-"
        if(node.right is not None):
            string_right=node.right.index
        print("index: ",node.index,"\tleft child: ",string_left,"\tright child: ",string_right,"\tpriority: ",node.priority,"\tsum: ",node.sum,"\tmax_path_to_leaf: ",node.max_path_to_leaf)
        if (node.left is not None):
            self.print(node.left)
        if (node.right is not None):
            self.print(node.right)



class Sum_tree_element():
    def __init__(self,value,priority,sum,father,left,right,index):
        self.value=value            # value that has to be saved
        self.priority=priority      # priority of this node that has to be summed
        self.sum=sum                # sum of priorities= sum of both sub trees and priority
        self.father=father          # father node
        self.left=left              # left child
        self.right=right            # right child
        self.index=index            # index, which is used for deleting nodes
        self.max_path_to_leaf=0
        self.min_path_to_leaf=0
        self.update_path_lengths()

    def update_path_lengths(self):
        max_path_to_leaf_left = 0
        min_path_to_leaf_left=0
        if self.left is not None:
            max_path_to_leaf_left=self.left.max_path_to_leaf
            min_path_to_leaf_left=self.left.min_path_to_leaf
        max_path_to_leaf_right = 0
        min_path_to_leaf_right = 0
        if self.right is not None:
            max_path_to_leaf_right=self.right.max_path_to_leaf
            min_path_to_leaf_right=self.right.min_path_to_leaf
        self.max_path_to_leaf=1+max(max_path_to_leaf_left,max_path_to_leaf_right)
        self.min_path_to_leaf=1+min(min_path_to_leaf_left,min_path_to_leaf_right)
