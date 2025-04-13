import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import copy 

import numpy as np
import torch
from agent.buffer import OfflineBuffer
from agent.encoder import Encoder 
from dataclasses import dataclass
from typing import Callable
import gymnasium as gym
import ogbench 
torch.backends.cudnn.benchmark = True
import torch.nn as nn 
import torch.nn.functional as F 

@dataclass
class Hyperparameters:
    # Generic
    batch_size: int = 256
    buffer_size: int = 1e6
    discount: float = 0.99
    target_update_rate: int = 250
    exploration_noise: float = 0.1
    
    # TD3
    target_policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    
    # LAP
    alpha: float = 0.4
    min_priority: float = 1
    
    # TD3+BC
    lmbda: float = 0.1
    
    # Checkpointing
    max_eps_when_checkpointing: int = 20
    steps_before_checkpointing: int = 75e4 
    reset_weight: float = 0.9
    
    # Encoder Model
    zs_dim: int = 256
    enc_hdim: int = 256
    enc_activ: Callable = F.elu
    encoder_lr: float = 3e-4
    
    # Critic Model
    critic_hdim: int = 256
    critic_activ: Callable = F.elu
    critic_lr: float = 3e-4
    
    # Actor Model
    actor_hdim: int = 256
    actor_activ: Callable = F.relu
    actor_lr: float = 3e-4

# Replay buffer
class OfflineWMBuffer(OfflineBuffer):
    """
    Specifically for Offline Data with goal conditioning 
    """
    def __init__(self, *args, **kwargs):
        super(OfflineWMBuffer, self).__init__(*args, **kwargs)
    
    def load_ogbench(self, dataset_path, weight=None):
        self.dataset = 'ogbench'
        self.weight = weight
        from agent.ogb_utils import load_dataset
        if "visual" in dataset_path:
            self.obs_type = 'pixel'
            self.normalize = 255.0
        else:
            self.obs_type = 'state'
            self.normalize = 1.0
        dataset = load_dataset(dataset_path, np.uint8)

        self.state = dataset['observations']
        self.action = dataset['actions']
        self.next_state = dataset['next_observations']
        # self.reward = None
        # self.not_done = 1 - dataset['terminals']
        self.size = np.prod(dataset['terminals'].shape)
        self.num_traj, self.traj_length = dataset['terminals'].shape 
        if self.prioritized:
            self.priority = torch.ones(self.size).to(self.device)

    def sample(self, batch_size, gc_negative=True, H=5):
        idxs = np.random.randint(0, self.num_traj, batch_size)
        traj_pos = np.random.randint(0, self.traj_length-1, batch_size)
        offset = np.random.geometric(p=1 - 0.99, size=batch_size) * np.where(np.random.rand(batch_size)<0.2, 0, 1)
        goal_pos = np.minimum(offset + traj_pos, self.traj_length-1)
        if self.weight == 'exp':
            weight = 0.99*(goal_pos -traj_pos) 
        elif self.weight == 'linear':
            weight = 1  - ((goal_pos - traj_pos) / ((self.traj_length-1) - traj_pos))
        else:
            weight = 1*(goal_pos - traj_pos)     
        assert not (len(np.where(weight < 0)[0]) > 0)
        success = goal_pos == traj_pos
        sub = 0.5 if self.obs_type == 'pixel' else 0
        batch = {}
        batch['observation'] = torch.tensor(self.state[idxs, traj_pos]/self.normalize - sub, dtype=torch.float, device=self.device)
        batch['next_observation'] = torch.tensor(self.next_state[idxs, traj_pos]/self.normalize - sub, dtype=torch.float, device=self.device)
        batch['action'] = torch.tensor(self.action[idxs, traj_pos], dtype=torch.float, device=self.device)
        batch['goal'] = torch.tensor(self.state[idxs, goal_pos]/self.normalize - sub, dtype=torch.float, device=self.device)
        success = success.astype(float)
        batch['not_done'] = torch.tensor(1 - success, dtype=torch.float, device=self.device)
        reward = success - (1.0 if gc_negative else 0.0)
        batch['reward'] = torch.tensor(reward, dtype=torch.float, device=self.device)
        batch['weight'] = torch.tensor(weight, dtype=torch.float, device=self.device)
        # Warmup
        if H > 0:
            time_indices = traj_pos[:, None] - np.arange(H)[::-1]
            batch['past_observation'] = torch.tensor(self.state[idxs[:,None], time_indices]/self.normalize - sub, dtype=torch.float, device=self.device)
            batch['past_action'] = torch.tensor(self.action[idxs[:,None], time_indices], dtype=torch.float, device=self.device)
            return (
                batch['observation'],
                batch['action'],
                batch['next_observation'],
                batch['reward'].unsqueeze(1),
                batch['not_done'].unsqueeze(1),
                batch['goal'],
                batch['weight'].unsqueeze(1),
                batch['past_observation'],
                batch['past_action']
            )
        
        return (
                batch['observation'],
                batch['action'],
                batch['next_observation'],
                batch['reward'].unsqueeze(1),
                batch['not_done'].unsqueeze(1),
                batch['goal'],
                batch['weight'].unsqueeze(1),
            )
# Value model V(Z(S,A))

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, 
                 zs_dim=256, hdim=256, activ=F.elu, 
                 goal=False, obs_type='pixel'):
        super(ValueNetwork, self).__init__()
        self.goal = goal
        self.activ = activ
        self.obs_type=obs_type
        if obs_type == 'pixel':
            C, _, _ = state_dim        
            self.encoder = Encoder(inp_channel=C, 
                                    filters=[32, 32, 32], 
                                    dropout=0.1, 
                                    image_size=state_dim, 
                                    batch_norm=False, 
                                    layer_norm=True, 
                                    out_emb = hdim)
        feature_layer = 1536 + hdim
        self.q1 = nn.Linear(feature_layer, hdim)    
        self.q2 = nn.Linear(hdim, hdim)
        self.q3 = nn.Linear(hdim, 1)


        self.q4 = nn.Linear(feature_layer, hdim)        
        self.q5 = nn.Linear(hdim, hdim)
        self.q6 = nn.Linear(hdim, 1)


    def forward(self, zsa, goal):
        
        goal = self.encoder(goal)        
        sa = torch.cat([zsa, goal], 1)
        
        q1 = AvgL1Norm(self.q1(sa))    
        q1 = self.activ(self.q2(q1))
        q1 = self.q3(q1)

        q2 = AvgL1Norm(self.q4(sa))
        q2 = self.activ(self.q5(q2))
        q2 = self.q6(q2)
        return torch.cat([q1, q2], 1)
# Critic Q(S, A)
class SharedEncoderCritic(nn.Module):
    def __init__(self, state_dim, action_dim, 
                 zs_dim=256, hdim=256, activ=F.elu, 
                 goal=False, obs_type='pixel'):
        super(SharedEncoderCritic, self).__init__()
        self.goal = goal
        self.activ = activ
        self.obs_type=obs_type
        if obs_type == 'pixel':
            C, _, _ = state_dim        
            self.encoder = Encoder(inp_channel=C, 
                                    filters=[32, 32, 32], 
                                    dropout=0.1, 
                                    image_size=state_dim, 
                                    batch_norm=False, 
                                    layer_norm=True, 
                                    out_emb = hdim)
            feature_layer = ((2*hdim) + action_dim) if goal else (hdim + action_dim) 
        else:
            feature_layer = ((2*state_dim) + action_dim) if goal else (state_dim + action_dim)
        
        self.q1 = nn.Linear(feature_layer, hdim)    
        self.q2 = nn.Linear(hdim, hdim)
        self.q3 = nn.Linear(hdim, 1)


        self.q4 = nn.Linear(feature_layer, hdim)        
        self.q5 = nn.Linear(hdim, hdim)
        self.q6 = nn.Linear(hdim, 1)


    def forward(self, state, action, goal):
        
        if self.obs_type == 'pixel':
            state = self.encoder(state)
            if self.goal:
                goal = self.encoder(goal)
            sa = torch.cat([state, goal, action], 1) if self.goal else torch.cat([state, action], 1)
        else:
            sa = torch.cat([state, goal, action], 1) if self.goal else torch.cat([state, action], 1)
        
        q1 = AvgL1Norm(self.q1(sa))    
        q1 = self.activ(self.q2(q1))
        q1 = self.q3(q1)

        q2 = AvgL1Norm(self.q4(sa))
        q2 = self.activ(self.q5(q2))
        q2 = self.q6(q2)
        return torch.cat([q1, q2], 1)
# Actor 1. Pi(Z) or 2. Pi(s, z) or 3. pi(s)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, 
                 zs_dim=256, hdim=256, activ=F.relu, 
                 goal=False, obs_type='pixel'):
        super(Actor, self).__init__()
        self.goal = goal
        self.activ = activ
        if obs_type == 'pixel':
            C, _, _ = state_dim        
            self.l0 = Encoder(inp_channel=C, 
                                filters=[32, 32, 32], 
                                dropout=0.1, 
                                image_size=state_dim, 
                                batch_norm=False, 
                                layer_norm=True, 
                                out_emb = hdim)
        else:
            self.l0 = nn.Linear(state_dim, hdim)
        
        feature_dim = 2*hdim if goal else hdim    
        self.l1 = nn.Linear(feature_dim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, action_dim)


    def forward(self, state, goal=None):
        if self.goal:
            goal = AvgL1Norm(self.l0(goal))
        a = AvgL1Norm(self.l0(state))
        a = torch.cat([a, goal], 1) if self.goal else a
        a = self.activ(self.l1(a))
        a = self.activ(self.l2(a))
        return torch.tanh(self.l3(a))
# Agent 

# loss functions 
def AvgL1Norm(x, eps=1e-8):
    return x/x.abs().mean(-1,keepdim=True).clamp(min=eps)

def LAP_huber(x, min_priority=1):
    return torch.where(x < min_priority, 0.5 * x.pow(2), min_priority * x).sum(1).mean()

class Agent(object):
    def __init__(self, state_dim, action_dim, max_action, wm,
                 offline=False, hp=Hyperparameters(), goal_cond=False, obs_type='pixel'): 
        # Changing hyperparameters example: hp=Hyperparameters(batch_size=128)
        self.gc = goal_cond
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hp = hp

        self.actor = Actor(state_dim, action_dim, hp.zs_dim, hp.actor_hdim, hp.actor_activ, True, obs_type).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=hp.actor_lr)
        self.actor_target = copy.deepcopy(self.actor)

        # self.critic = Critic(state_dim, action_dim, hp.zs_dim, hp.critic_hdim, hp.critic_activ).to(self.device)
        self.critic = SharedEncoderCritic(state_dim, action_dim, hp.zs_dim, hp.critic_hdim, hp.critic_activ, True, obs_type).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=hp.critic_lr)
        self.critic_target = copy.deepcopy(self.critic)

        self.value = ValueNetwork(state_dim, action_dim, hp.zs_dim, hp.critic_hdim, hp.critic_activ, True, obs_type).to(self.device)
        self.value_optimizer = torch.optim.Adam(self.critic.parameters(), lr=hp.critic_lr)

        self.checkpoint_actor = copy.deepcopy(self.actor)

        self.replay_buffer = OfflineWMBuffer(state_dim, action_dim, self.device, hp.buffer_size, hp.batch_size, 
            max_action, normalize_actions=True, prioritized=False, obs_type=obs_type, offline=offline)
        self.wm = wm
        self.max_action = max_action
        self.offline = offline
        self.obs_type = obs_type
        self.training_steps = 0

        # Checkpointing tracked values
        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.max_eps_before_update = 1
        self.min_return = 1e8
        self.best_min_return = -1e8

        # Value clipping tracked values
        self.max = -1e8
        self.min = 1e8
        self.max_target = 0
        self.min_target = 0


    def select_action(self, state, goal=None, use_checkpoint=False, use_exploration=True):
        with torch.no_grad():
            if self.obs_type == 'pixel':
                state = torch.tensor(state.copy().transpose(2, 0, 1), dtype=torch.float, device=self.device).unsqueeze(0)/255.0 - 0.5
                goal = (torch.tensor(goal.copy().transpose(2, 0, 1), dtype=torch.float, device=self.device).unsqueeze(0)/255.0 - 0.5) if self.gc else None
            else:
                state = torch.tensor(state.reshape(1,-1), dtype=torch.float, device=self.device)
                goal = torch.tensor(goal.reshape(1, -1), dtype=torch.float, device=self.device) if self.gc else None
            
            action = self.actor(state, goal) 

            return action.clamp(-1,1).cpu().data.numpy().flatten() * self.max_action
        
    def train(self):
        metrics = {}
        self.training_steps += 1
        if self.gc:
            state, action, next_state, reward, not_done, goal, weight, past_state, past_action = self.replay_buffer.sample(self.hp.batch_size, H=5)
        else:
            state, action, next_state, reward, not_done = self.replay_buffer.sample(self.hp.batch_size)
            goal = None


        #########################
        # Update Value_function
        #########################
        with torch.no_grad():
            Q_value = self.critic_target(state, action, goal).min(1, keepdim=True)[0]
            zsa = self.wm.Zsa(past_state, past_action).detach() 
        v = self.value(zsa, goal)
        value_loss = LAP_huber(v -  Q_value)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        metrics.update({
            "train/Value_loss": value_loss.item(),
            'train/v_mean': v.mean().item(),
            'train/v_max': v.max().item(),
            'train/v_min': v.min().item(),
        })
        
        #########################
        # Update Critic
        #########################
        with torch.no_grad():
            # fixed_target_zs = self.fixed_encoder_target.zs(next_state)
        
            if self.offline:
                noise = 0.0
            else:
                noise = (torch.randn_like(action) * self.hp.target_policy_noise).clamp(-self.hp.noise_clip, self.hp.noise_clip)
            next_action = (self.actor_target(next_state, goal) + noise).clamp(-1,1)
            next_zsa = self.wm.Zsa(past_state, past_action, next_state, next_action).detach() # here we already have the deter
            next_v = self.value(next_zsa, goal)
            Q_target = reward + not_done * self.hp.discount * next_v # .clamp(self.min_target, self.max_target) 
                
            self.max = max(self.max, float(Q_target.max()))
            self.min = min(self.min, float(Q_target.min()))
        
        Q = self.critic(state, action, goal)
        td_loss = (Q - Q_target).abs()
        critic_loss = LAP_huber(td_loss)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        metrics.update({
            'train/critic_loss': critic_loss.item(),
            'train/critic_mean': Q.mean().item(),
            'train/critic_max': Q.max().item(),
            'train/critic_min': Q.min().item(),
        })
        #########################
        # Update Actor
        #########################
        if self.training_steps % self.hp.policy_freq == 0:
            actor = self.actor(state, goal)
            past_action[:, -1] = actor
            zsa = self.wm.Zsa(past_state, past_action).detach() 
            Q = self.value(zsa, goal)
            actor_loss = -Q.mean()
            if self.offline:
                # actor_loss = weight * (actor_loss + self.hp.lmbda * Q.abs().mean().detach() * F.mse_loss(actor, action))
                weighted_qval = (weight * -Q).mean()
                offline_coef = (weight * Q).abs().mean().detach()
                bc_loss = self.hp.lmbda * offline_coef * F.mse_loss(actor , action)
                actor_loss = weighted_qval + bc_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            metrics.update({
                'train/actor_loss': actor_loss.item(),
                'train/actor_qval': weighted_qval.item(),
                'train/bc_loss': bc_loss.item(),
                'train/alpha':offline_coef.item(),
                'train/actor_qmean': Q.mean().item(),
            })
        
        #########################
        # Update Iteration
        #########################
        if self.training_steps % self.hp.target_update_rate == 0:
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
            # self.fixed_encoder_target.load_state_dict(self.fixed_encoder.state_dict())
            # self.fixed_encoder.load_state_dict(self.encoder.state_dict())
            
            self.replay_buffer.reset_max_priority()

            self.max_target = self.max
            self.min_target = self.min
            metrics.update({
                "train/min_target": self.min_target, 
                "train/max_target": self.max_target, 
            })
            return True, metrics
        return False, metrics

    def save(self, save_folder):
        # Save models/optimizers
        models = [
            'critic', 'critic_target', 'critic_optimizer',
            'actor', 'actor_target', 'actor_optimizer',
            'value', 'value_optimizer'
        ]
        save_folder = save_folder / str(self.training_steps)
        self.make_dir(save_folder)
        for k in models: torch.save(self.__dict__[k].state_dict(), save_folder / f'{k}.pt')

        # Save variables
        vars = ['hp']
        var_dict = {k: self.__dict__[k] for k in vars}
        np.save(f'{save_folder}/agent_var.npy', var_dict)

    def load(self, save_folder):
        # Load models/optimizers.
        models = [
            'critic', 'critic_target', 'critic_optimizer',
            'actor', 'actor_target', 'actor_optimizer',
            'value', 'value_optimizer'
        ]
        for k in models: self.__dict__[k].load_state_dict(torch.load(f'{save_folder}/{k}.pt', weights_only=True))

        # Load variables.
        var_dict = np.load(f'{save_folder}/agent_var.npy', allow_pickle=True).item()
        for k, v in var_dict.items(): self.__dict__[k] = v

    
    def make_dir(self, path):
        if not os.path.exists(path):
            os.mkdir(path) 
        

