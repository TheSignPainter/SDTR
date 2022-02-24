#!/usr/bin/env python
# coding: utf-8


import os, sys
import time
sys.path.insert(0, '..')
import numpy as np
import matplotlib.pyplot as plt
from lib import Dataset
import torch, torch.nn as nn
import torch.nn.functional as F
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

exps = 1
seeds = [random.randint(0, 100000) for _ in range(exps)]

data = Dataset("SARCOS", random_state=seeds[0], quantile_transform=True, quantile_noise=1e-3)
in_features = data.X_train.shape[1]
# print(in_features)

mu, std = data.y_train.mean(), data.y_train.std()
normalize = lambda x: ((x - mu) / std).astype(np.float32)
data.y_train, data.y_valid, data.y_test = map(normalize, [data.y_train, data.y_valid, data.y_test])


def run_exp(n_tree, depth, lmbda, lmbda2, use_reweight, use_hidden):
    losses = []
    ent = []
    for ex in range(exps):
    
        experiment_name = 'SARCOS_SDTR'
        experiment_name = '{}_{}.{:0>2d}.{:0>2d}_{:0>2d}:{:0>2d}:{:0>2d}_exp{}'.format(experiment_name, *time.gmtime()[:6], ex)
        print("experiment:", experiment_name)
        
        # print("mean = %.5f, std = %.5f" % (mu, std))

        from lib import DenseBlockSDTR, SDTR, Lambda

        class DBSDTR(nn.Module):
            
            def __init__(self):
                super(DBSDTR, self).__init__()
                self.DBS = DenseBlockSDTR(in_features, n_tree, num_layers=1, tree_dim=3, depth=depth, flatten_output=False,
                           Module=SDTR, hidden_dim=in_features if use_hidden else None, init_func="normal", lmbda=lmbda, lmbda2=lmbda2)
                self.output_avg = Lambda(lambda x: x[..., 0].mean(dim=-1))
            
            def forward(self, x):
                output, reg, l1 = self.DBS(x)
                output = self.output_avg(output)
                return output, reg, l1

            def eval_entropy(self):
                return self.DBS.eval_entropy()
        
        model = DBSDTR().to(device, non_blocking=True)
        print("Model built.")
        print("# parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
        with torch.no_grad():
            res = model(torch.as_tensor(data.X_train[:128], device=device))
            # trigger data-aware init
            
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        lr = 1e-4
        from qhoptim.pyt import QHAdam
        params_to_opt = None
        if use_reweight:
            params = []
            responses = []
            for name, param in model.named_parameters():
                if "response" in name:
                    responses.append(param)
                else:
                    params.append(param)
                    params_to_opt = [{"params": params}, {"params": responses, "lr": 2**depth*lr}]
        optimizer_params = {'lr': lr, 'nus':(0.8, 0.92), 'betas':(0.85, 0.97) }

        import lib
        trainer = lib.Trainer(
            model=model, loss_function=F.mse_loss,
            experiment_name=experiment_name,
            warm_start=False,
            Optimizer=QHAdam,
            optimizer_params=optimizer_params,
            params_to_opt=params_to_opt,
            verbose=False,
            n_last_checkpoints=2
        )


        mse_history = []
        best_mse = float('inf')
        best_step_mse = 0
        early_stopping_rounds = 2000
        report_frequency = 100 
        epochs = 10000

        for batch in lib.iterate_minibatches(data.X_train, data.y_train, batch_size=4096, 
                                                        shuffle=True, epochs=epochs):

            metrics = trainer.train_on_batch(*batch, device=device)
            
            if trainer.step % report_frequency == 0:
                trainer.save_checkpoint()
                # print("Step: %d / %d, "%(trainer.step, epochs), end=" ")
                # for key, val in metrics.items():
                #     print("%s:%.3f |"%(key,val), end=" ")
                # print("ent: %.3f"%trainer.evaluate_entropy())
                # print()
                # # trainer.average_checkpoints(out_tag='avg')
                # trainer.load_checkpoint(tag='avg')
                mse = trainer.evaluate_mse(
                    data.X_valid, data.y_valid, device=device, batch_size=4096)

                if mse < best_mse:
                    best_mse = mse
                    best_step_mse = trainer.step
                    trainer.save_checkpoint(tag='best_mse')
                mse_history.append(mse)
                # trainer.load_checkpoint()  # last
                trainer.remove_old_temp_checkpoints()

#                 print("[%d] Loss %.5f" % (trainer.step, metrics['loss']))
#                 print("Val MSE: %0.5f" % (mse), flush=True)
                if trainer.step > best_step_mse + early_stopping_rounds:
                    # print('BREAK. There is no improvment for {} steps'.format(early_stopping_rounds))
                    print("Best step: ", best_step_mse)
                    print("Best Val MSE: %0.5f" % (best_mse))
                    break

        # print("Training done.")

        trainer.load_checkpoint(tag='best_mse')
        mse = trainer.evaluate_mse(data.X_test, data.y_test, device=device, batch_size=4096)
        # print("Test MSE: %0.5f" % (mse))

        losses.append(mse * std ** 2)
        ent.append(model.eval_entropy())
    print("=========================================")
    print("N_tree:", n_tree, "Depth:", depth, "| Lmbda:", lmbda, "| Lmbda2:", lmbda2, "| Use_hidden:", use_hidden, "| Use_reweight:", use_reweight)
    losses = np.array(losses)
    ent = np.array(ent)
    print("Mean:", losses.mean(), "| std:", losses.std(), "| ent:", ent.mean())
    print()

depths = [8]
lmbdas = [0.01]
lmbdas2 = [0.01]
# lmbdas2 = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0]
n_trees = [64]
use_hidden=[False]
use_reweight=[True]

for nt in n_trees:
    for d in depths:
        for l in lmbdas:
            for l2 in lmbdas2:
                for ur in use_reweight:
                    for uh in use_hidden:
                        run_exp(nt, d, l, l2, ur, uh)
