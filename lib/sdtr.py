
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

from .nn_utils import sparsemax, sparsemoid, ModuleWithInit
from .utils import check_numpy
from warnings import warn


class SDTR(ModuleWithInit):
    
    def __init__(self, in_features, num_trees, depth=6, tree_dim=1, flatten_output=True,
                 init_func="zero", hidden_dim=256, lmbda=0.1, lmbda2=0.01, **kwargs
                 ):
        """

        """
        super().__init__()
        self.depth, self.num_trees, self.tree_dim, self.flatten_output = depth, num_trees, tree_dim, flatten_output
        self.in_features, self.hidden_dim, self.lmbda, self.lmbda2 = in_features, hidden_dim, lmbda, lmbda2

        self.response = nn.Parameter(torch.zeros([num_trees, tree_dim, 2 ** depth]), requires_grad=True)
        #initialize_response_(self.response)
        self.init_func = init_func
        in_dim = in_features
        if self.hidden_dim:
            self.input_fc = torch.nn.Linear(in_features, out_features=self.hidden_dim)
            in_dim = hidden_dim
        beta = 1.5
        self.feature_logit_layers = []
        self.betas = []
        for cur_depth in range(depth):
            self.feature_logit_layers.append(nn.Linear(in_features, num_trees*(2**cur_depth)))
            torch.nn.init.xavier_uniform_(self.feature_logit_layers[-1].weight)
            beta_array = np.full((num_trees, 2**cur_depth), beta)
            self.betas.append(nn.Parameter(torch.FloatTensor(beta_array), requires_grad=True))
        self.feature_logit_layers = nn.ModuleList(self.feature_logit_layers)
        self.betas = nn.ParameterList(self.betas)
        
    def forward(self, input):
        assert len(input.shape) >= 2
        if len(input.shape) > 2:
            return self.forward(input.view(-1, input.shape[-1])).view(*input.shape[:-1], -1)
        batch_size = input.size()[0]
        # new input shape: [batch_size, in_features]
        device = next(self.parameters()).device 
#        print("device:", device)
        # print("batch size:", batch_size)
        if self.hidden_dim:
            features = self.input_fc(input)
        else:
            features = input
        reg_loss = 0.
        l1_loss = None
        reweighting = 1
        # device = next(self.parameters()).device
        path_prob_prev = Variable(torch.ones(batch_size, self.num_trees, 1), requires_grad=True).to(device)
        depth = len(self.feature_logit_layers)
#        print("prev device:", path_prob_prev.type())
        for cur_depth in range(depth):
 #           print("layer device:", self.feature_logit_layers[cur_depth].weight.type())
            current_prob_logit = self.feature_logit_layers[cur_depth](input).view(batch_size, self.num_trees, -1)
            # [batch_size, num_trees, 2**cur_depth]
            for param in self.feature_logit_layers[cur_depth].parameters():
                if l1_loss is None:
                    l1_loss = param.norm(1)/self.num_trees * reweighting * self.lmbda2
                else:
                    l1_loss += param.norm(1)/self.num_trees * reweighting * self.lmbda2
            reweighting /= 2
            current_prob = torch.sigmoid(torch.einsum("bnc, nc->bnc", current_prob_logit, self.betas[cur_depth]))\
            .view(batch_size, self.num_trees, 2**cur_depth)
            # [batch_size, num_trees, 2**cur_depth]
            current_prob_right = 1-current_prob
            # [batch_size, num_trees, 2**cur_depth]
            penalty = torch.einsum("bnc, bnc->nc", current_prob, path_prob_prev) / (torch.einsum("ijk->jk", path_prob_prev)+1e-4)
            # summing the example penalties in each batches.
            # [1, num_trees, 2**cur_depth]
            reg_loss -= self.lmbda * 2 ** (-cur_depth) * 0.5 *torch.mean((torch.log(penalty) + torch.log(1-penalty)))
            current_prob = current_prob * path_prob_prev
            current_prob_right = current_prob_right * path_prob_prev
            path_prob_prev = torch.stack((current_prob.unsqueeze(-1), current_prob_right.unsqueeze(-1)), dim=3)\
                .view(batch_size, self.num_trees, 2**(cur_depth+1))
 
        response_weights = path_prob_prev
        # print(torch.max(response_weights))
        # print(response_weights.shape, self.response.shape)
        response = torch.einsum('bnd,ncd->bnc', response_weights, self.response)
        response = response.flatten(1, 2) if self.flatten_output else response
        ####
        # del features, leaf_probs, response_weights, self.path_prob_init
        return response, reg_loss, l1_loss

    def __repr__(self):
        return "{}(in_features={}, num_trees={}, depth={}, tree_dim={}, flatten_output={})".format(
            self.__class__.__name__, self.feature_selection_logits.shape[0],
            self.num_trees, self.depth, self.tree_dim, self.flatten_output
        )


    def initialize(self, input, eps=1e-6):
        # print("Initialized by %s" % self.init_func)
        # try random initialization first.
        if self.init_func == "uniform":
            torch.nn.init.uniform_(self.response,-1, 1)
        elif self.init_func == "xuniform":
            torch.nn.init.xavier_uniform_(self.response)
        if self.init_func == "zero":
            pass
        if self.init_func == "normal":
            torch.nn.init.normal_(self.response)

    def get_entropy(self):
        eps=1e-8
        total_entropy = 0
        with torch.no_grad():
            depth = len(self.feature_logit_layers)
            for cur_depth in range(depth):
                param = self.feature_logit_layers[cur_depth].weight # [NUM_Trees*2**cur_depth, x]
                assert len(param) == self.num_trees * 2**cur_depth
                param = param.view([self.num_trees, 2**cur_depth, -1])
                for tree in range(len(param)):
                    for node in range(2**cur_depth):
                        W = param[tree, node, :].abs()
                        W += eps
                        W /= W.sum()
                        total_entropy -= (W * torch.log(W)* 2**(-cur_depth)).sum().item()/len(param) # Avg. entropy of trees.
        return total_entropy
            

    def binning(self, input):
        """
        Given a (large) input batch, return a list [self.depth**2].
        Valued by the probable bins.
        """
        min_data_in_bin = 2
        total_sample_cnt = input.shape[0]
        input_ys = list(input[:, -1].cpu().numpy().tolist())
        from collections import Counter
        v_counter = Counter(input_ys)
        sorted_keys = sorted(list(v_counter.keys()))
        min_val = sorted_keys[0]
        max_val = sorted_keys[-1]
        max_bin = 2**self.depth
        mean_bin_size = total_sample_cnt // max_bin
        rest_bin_cnt = max_bin
        rest_sample_cnt = total_sample_cnt
        is_big_bin = [False] * len(sorted_keys)
        for ind,k in enumerate(sorted_keys):
            if v_counter[k] >= mean_bin_size:
                is_big_bin[ind] = True
                rest_bin_cnt -= 1
                rest_sample_cnt -= v_counter[k]
        mean_bin_size = rest_sample_cnt // rest_bin_cnt
        ubounds = [0] * max_bin
        lbounds = [0] * max_bin
        

        


