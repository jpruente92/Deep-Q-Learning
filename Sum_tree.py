import random


from collections import deque

# in the sum tree the elements are saved
# in the queue the node are saved so that they can be removed when max size is reached
class Sum_tree_queue():
    def __init__(self,buffer_size,seed):
        self.sum_tree= Sum_tree(seed)
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
    def __init__(self,seed):
        self.root=None
        random.seed(seed)


    # works for trees and sub trees
    def add(self,new_node):
        new_node.father=None
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
        ancestor=father
        while True:
            ancestor.sum+=new_node.sum
            ancestor.update_path_lengths()
            if(ancestor.father is None):
                break
            ancestor=ancestor.father


    def add_new_value(self,value,priority,index):
        self.add(Sum_tree_element(value,priority,priority,None,None,None,index))


    # idea: remove the node and add all children to the tree via add
    # if node is root and has two children add one child to the other
    def remove(self,node):
        if self.root is node:
            if(node.left is None):
                if (node.right is None):
                    self.root=None
                    return
                else:
                    self.root=node.right
                    self.root.father=None
            else:
                if (node.right is None):
                    self.root=node.left
                    self.root.father=None
                else:
                    # tree with higher max path length will be the root
                    if(node.right.max_path_to_leaf>node.left.max_path_to_leaf):
                        self.root=node.right
                        self.root.father=None
                        self.add(node.left)
                    else:
                        self.root=node.left
                        self.root.father=None
                        self.add(node.right)
        else:
            father = node.father
            # delete node from father
            if father.left is not None and father.left.index == node.index:
                father.left=None
            elif father.right is not None and father.right.index == node.index:
                father.right=None
            else:
                assert False
            # subtract sum from all ancestors of node and update path lengths
            ancestor = father
            while True:
                ancestor.sum-=node.sum
                ancestor.update_path_lengths()
                if(ancestor.father is None):
                    break
                ancestor=ancestor.father
            # add children to tree
            if(node.left is not None):
                node.left.father=None
                self.add(node.left)
            if(node.right is not None):
                node.right.father=None
                self.add(node.right)

    def update_priority(self,node,new_priority):
        diff = node.priority-new_priority
        node.priority=new_priority
        current=node
        while current is not None:
            current.sum-=diff
            current=current.father


    def sample_values(self,replace,batch_size):
        nodes=[]
        priorities=[]
        for x in range(batch_size):
            random_number = random.random()
            node =self.root
            total_sum=self.root.sum
            while True:
                sum_left=0
                if(node.left is not None):
                    sum_left=node.left.sum
                if(node.left is not None and random_number<=sum_left/total_sum):
                    node=node.left
                elif(node.right is not None and random_number>=(sum_left+node.priority)/total_sum):
                    node=node.right
                    random_number-=((sum_left+node.priority)/total_sum)
                else:
                    nodes.append(node)
                    priorities.append(node.priority)
                    if not replace:
                        self.remove(node)
                    break
        # add all nodes back to the tree
        if not replace:
            for node in nodes:
                node.father=None
                node.left=None
                node.right=None
                node.sum=node.priority
                self.add(node)
        # divide priorities with total sum to get the probability for a single draw
        probabilities=[p/self.root.sum for p in priorities]
        return nodes,probabilities

    def consistency_check(self,node):
        if(self.root is node):
            assert (node.father is None)
        sum=node.priority
        if(node.left is not None):
            assert (node.left.father is node)
            self.consistency_check(node.left)
            sum+=node.left.sum
        if(node.right is not None):
            assert (node.right.father is node)
            self.consistency_check(node.right)
            sum+=node.right.sum
        assert(abs(sum-node.sum)<0.01)

    def print(self,node):
        string_father="-"
        if(node.father is not None):
            string_father=node.father.index
        string_left="-"
        if(node.left is not None):
            string_left=node.left.index
        string_right="-"
        if(node.right is not None):
            string_right=node.right.index
        print("index: ",node.index,"\tfather: ",string_father,"\tleft child: ",string_left,"\tright child: ",string_right,"\tpriority: ",node.priority,"\tsum: ",node.sum,"\tmax_path_to_leaf: ",node.max_path_to_leaf)
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
