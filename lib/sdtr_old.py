
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

from .nn_utils import sparsemax, sparsemoid, ModuleWithInit
from .utils import check_numpy
from warnings import warn

class InnerNode():

        def __init__(self, cur_depth, max_depth, input_dim, output_dim, lmbda, index=0):
            self.index = index
            self.max_depth = max_depth
            self.fc = nn.Linear(input_dim, 1)
            beta = torch.full([1], 2)
            self.beta = nn.Parameter(beta, requires_grad=True)
            self.leaf = False
            self.prob = None
            self.lmbda = lmbda * 2 ** (-cur_depth)
            self.build_child(cur_depth, max_depth, input_dim, output_dim, lmbda)

        def build_child(self, cur_depth, max_depth, input_dim, output_dim,lmbda):
            if cur_depth < self.max_depth:
                self.left = InnerNode(cur_depth+1, max_depth, input_dim, output_dim, lmbda, index=self.index*2)
                self.right = InnerNode(cur_depth+1, max_depth, input_dim, output_dim, lmbda, index=self.index*2 + 1)
            else :
                self.left = LeafNode(output_dim, index=self.index*2)
                self.right = LeafNode(output_dim, index=self.index*2 + 1)

        def forward(self, x):
            return(F.sigmoid(self.beta*self.fc(x)))
        
        def select_next(self, x):
            prob = self.forward(x)
            if prob < 0.5:
                return(self.left, prob)
            else:
                return(self.right, prob)
        def cal_prob_and_reg(self, x, path_prob):
            prob = self.forward(x) #probability of selecting right node
            # [batch_size, 1]
            penalty = torch.sum(prob * path_prob) / torch.sum(path_prob)
            reg_loss = -self.lmbda * 0.5 *(torch.log(penalty) + torch.log(1-penalty))
            left_leaf_accumulator, reg_left = self.left.cal_prob_and_reg(x, path_prob * (1-prob))
            right_leaf_accumulator, reg_right = self.right.cal_prob_and_reg(x, path_prob * prob)
            leaf_accumulator = torch.cat((left_leaf_accumulator, right_leaf_accumulator), dim=1)
            reg_loss += (reg_left+reg_right)
            return (leaf_accumulator, reg_loss)

class LeafNode():
    def __init__(self, output_dim=1, index= 0):
        param = torch.randn(output_dim)
        self.index = index
        # self.param = nn.Parameter(param, requires_grad=True)
        # nn.init.normal_(self.param)
        self.leaf = True

    def forward(self):
        return(self.param)

    def reset(self):
        pass

    def cal_prob_and_reg(self, x, path_prob):
        return (path_prob, 0)



class SDTROLD(ModuleWithInit):

    
    def __init__(self, in_features, num_trees, depth=6, tree_dim=1, flatten_output=True,
                 choice_function=sparsemax, bin_function=sparsemoid,
                 initialize_response_=nn.init.normal_, initialize_selection_logits_=nn.init.uniform_,
                 threshold_init_beta=1.0, threshold_init_cutoff=1.0, hidden_dim=256, lmbda=0.1, **kwargs
                 ):
        """

        """
        super().__init__()
        self.depth, self.num_trees, self.tree_dim, self.flatten_output = depth, num_trees, tree_dim, flatten_output
        self.in_features, self.hidden_dim, self.lmbda = in_features, hidden_dim, lmbda
        self.choice_function, self.bin_function = choice_function, bin_function

        self.response = nn.Parameter(torch.zeros([num_trees, tree_dim, 2 ** depth]), requires_grad=True)
        initialize_response_(self.response)

        self.input_fc = torch.nn.Linear(in_features, out_features=hidden_dim)
        self.log_temperatures = nn.Parameter(
            torch.full([num_trees, depth], float('nan'), dtype=torch.float32), requires_grad=True
        )
        self.roots = []
        for i in range(self.num_trees):
            self.roots.append(InnerNode(1, depth, self.hidden_dim, self.tree_dim, self.lmbda))
        self.collect_parameters() ##collect parameters and modules under root node
        # print(self.leaf_param_dict)
        # for key in self.leaf_param_dict:
        #     print(key, self.leaf_param_dict[key])

        # binary codes for mapping between 1-hot vectors and bin indices
        # with torch.no_grad():
        #     indices = torch.arange(2 ** self.depth)
        #     offsets = 2 ** torch.arange(self.depth)
        #     bin_codes = (indices.view(1, -1) // offsets.view(-1, 1) % 2).to(torch.float32)
        #     bin_codes_1hot = torch.stack([bin_codes, 1.0 - bin_codes], dim=-1)
        #     self.bin_codes_1hot = nn.Parameter(bin_codes_1hot, requires_grad=False)
        #     # ^-- [depth, 2 ** depth, 2]
    def forward(self, input):
        assert len(input.shape) >= 2
        if len(input.shape) > 2:
            return self.forward(input.view(-1, input.shape[-1])).view(*input.shape[:-1], -1)
        batch_size = input.size()[0]
        # print("batch size:", batch_size)
        features = self.input_fc(input)
        leaf_probs = []
        reg_loss = 0.
        device = next(self.parameters()).device
        self.path_prob_init = Variable(torch.ones(batch_size, 1), requires_grad=False).to(device)
        for root in self.roots:
            leaf_accumulator, reg_loss = root.cal_prob_and_reg(features, self.path_prob_init)
            # [batch_size, 2**depth]
            leaf_probs.append(leaf_accumulator.view(batch_size, 1, -1))
            reg_loss += reg_loss
        response_weights = torch.cat(leaf_probs, dim=1)
        # print(response_weights.shape, self.response.shape)
        response = torch.einsum('bnd,ncd->bnc', response_weights, self.response)
        response = response.flatten(1, 2) if self.flatten_output else response
        ####
        del features, leaf_probs, response_weights, self.path_prob_init
        return response, reg_loss

    def __repr__(self):
        return "{}(in_features={}, num_trees={}, depth={}, tree_dim={}, flatten_output={})".format(
            self.__class__.__name__, self.feature_selection_logits.shape[0],
            self.num_trees, self.depth, self.tree_dim, self.flatten_output
        )

    def initialize(self, input, eps=1e-6):
        # There is no need to initialize according to percentile.
        pass
    def collect_parameters(self):
        self.module_list = nn.ModuleList()
        self.param_list = nn.ParameterList()
        self.leaf_param_dict = nn.ParameterDict()
        # self.module_list.append(self.input_fc)
        # self.param_list.append(self.input_fc.param)
        for root in self.roots:
            nodes = [root]
            while nodes:
                node = nodes.pop(0)
                if node.leaf:
                    pass
                else:
                    fc = node.fc
                    beta = node.beta
                    nodes.append(node.right)
                    nodes.append(node.left)
                    self.param_list.append(beta)
                    self.module_list.append(fc)
