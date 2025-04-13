import torch.nn as nn
import torch

import tools.utils as utils
import agent.dreamer_utils as common
from collections import OrderedDict
import numpy as np

from tools.genrl_utils import *

def stop_gradient(x):
  return x.detach()

Module = nn.Module 
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
    
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512, max_action=1):
        super(Actor, self).__init__() 
        
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action
        self.apply(weights_init_)

    def forward(self, x, goal=None):
        x = F.silu(self.l1(x))
        x = F.silu(self.l2(x))
        x = torch.tanh(self.l3(x))
        return self.max_action * x
    
class BCPolicy(Module):
  def __init__(self, config, act_spec, feat_size, name='', embed_dim=1536, goal_condition=None):
    super().__init__()
    self.name = name
    self.cfg = config
    self.act_spec = act_spec
    self._use_amp = (config.precision == 16)
    self.device = config.device
    self.embed_dim = embed_dim
    self.gc = goal_condition
    inp_size = feat_size + embed_dim if goal_condition else 0
    
    if getattr(self.cfg, 'discrete_actions', False):
      self.cfg.actor.dist = 'onehot'

    self.actor_grad = getattr(self.cfg, f'{self.name}_actor_grad'.strip('_'))
    
    self.inp_size = feat_size + embed_dim
    self.acs_space = act_spec.shape[0]
    self.actor = common.MLP(self.inp_size, self.acs_space, **self.cfg.actor)
    self.actor_opt = common.Optimizer('actor', self.actor.parameters(), **self.cfg.actor_opt, use_amp=self._use_amp)
    # different
    # self.actor2 = Actor(inp_size, act_spec.shape[0])
    # self.actor_opt2 = common.Optimizer('actor2', self.actor2.parameters(), **self.cfg.actor_opt, use_amp=self._use_amp)
  def reset_actor(self):
      self.actor = Actor(2*1536, 8).to(self.device)
      self.actor_opt = common.Optimizer('actor', self.actor.parameters(), **self.cfg.actor_opt, use_amp=self._use_amp)
  '''
  def update(self, world_model, start, batch, task_cond=None):
    metrics = {}
    with common.RequiresGrad(self.actor):
        # with torch.cuda.amp.autocast(enabled=self._use_amp):
        with torch.amp.autocast('cuda', enabled=self._use_amp):
            # action, entropy = self.OnestepBC(world_model, start, task_cond)
            action_dist = self.OnestepBC(world_model, start, task_cond)
            acs = batch['action'].reshape(-1, batch['action'].shape[-1]).detach()
            actor_loss = -1 * action_dist.log_prob(acs).mean()
            # actor_loss += 1.0 * ((acs - action_dist.mean)**2).mean()
            metrics.update(self.actor_opt(actor_loss, self.actor.parameters()))
            # with torch.no_grad():
            #   acs = action_dist.mean.detach()
            #   mse_loss = ((batch['action'].reshape(-1, batch['action'].shape[-1]) - acs)**2).mean().item()
            #   min_value, mean_value, max_value = acs.min().item(), acs.mean().item(), acs.max().item()
            #   metrics.update({"MSE_loss" : mse_loss}) 
            #   metrics.update({"min_action" : min_value}) 
            #   metrics.update({"mean_action" : mean_value}) 
            #   metrics.update({"max_action" : max_value}) 
            # metrics.update({"Policy_MSE" : mse_loss.item()actor_loss, "Policy_Entropy" : entropy.item()})   
            metrics.update({"Log Prob" : actor_loss})   
    return { f'{self.name}_{k}'.strip('_') : v for k,v in metrics.items() }
    
  def OnestepBC(self, world_model, start, task_cond=None, eval_policy=False):
    flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
    start = {k: flatten(v) for k, v in start.items()}
    start['feat'] = world_model.rssm.get_feat(start)
    inp = start['feat'] if task_cond is None else torch.cat([start['feat'], task_cond], dim=-1)
    policy_dist = self.actor(stop_gradient(inp))
    return policy_dist
  '''
  def update(self, world_model, start, batch, task_cond=None):
    metrics = {}
    with common.RequiresGrad(self.actor):
        with torch.amp.autocast('cuda', enabled=self._use_amp):
            acs = self.SimpleBC(world_model, start, task_cond)
            action = batch['action'].reshape(-1, batch['action'].shape[-1])
            loss = F.mse_loss(acs, action)
            metrics.update(self.actor_opt(loss, self.actor.parameters()))
            metrics.update({"MSE_loss" : loss})   
    return { f'{self.name}_{k}'.strip('_') : v for k,v in metrics.items() }
  
  def SimpleBC(self, world_model, start, task_cond=None, eval_policy=False):
    flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
    start = {k: flatten(v) for k, v in start.items()}
    start['feat'] = world_model.rssm.get_feat(start)
    inp = start['feat'] if task_cond is None else torch.cat([start['feat'], task_cond], dim=-1)
    acs = self.actor(stop_gradient(inp))
    return acs