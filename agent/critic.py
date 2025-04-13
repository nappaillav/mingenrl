import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from .encoder import Encoder
from .utils import AvgL1Norm

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.encoder1 = Encoder(inp_channel=3, 
                               filters=[32, 32, 32], 
                               dropout=0.1, 
                               image_size=(3, 64, 64), 
                               batch_norm=False, 
                               layer_norm=True, 
                               out_emb = hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        state = self.encoder1(state)
        x = F.relu(self.linear1(state))
        x = self.linear2(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()
        # input_image assumed to be 3x64x64
        # Q1 architecture
        self.encoder1 = Encoder(inp_channel=3, 
                               filters=[32, 32, 32], 
                               dropout=0.1, 
                               image_size=(3, 64, 64), 
                               batch_norm=False, 
                               layer_norm=True, 
                               out_emb = hidden_dim)
        self.linear1 = nn.Linear(hidden_dim + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.encoder2 = Encoder(inp_channel=3, 
                               filters=[32, 32, 32], 
                               dropout=0.1, 
                               image_size=(3, 64, 64), 
                               batch_norm=False, 
                               layer_norm=True, 
                               out_emb = hidden_dim)
        self.linear3 = nn.Linear(hidden_dim+num_actions, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        
        state1 = self.encoder1(state)
        state2 = self.encoder2(state)

        state1 = torch.cat([state1, action], 1)
        state2 = torch.cat([state2, action], 1)

        x1 = F.relu(self.linear1(state1))
        x1 = self.linear2(x1)
        
        x2 = F.relu(self.linear3(state2))
        x2 = self.linear4(x2)

        return x1, x2

# TD3
class Critic(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim, goal=None):
        super(Critic, self).__init__() 
        self.goal = goal
        if len(state_dim) == 3:
            C, _,_ = state_dim
            # Q1 
            self.encoder1 = Encoder(inp_channel=C, 
                                    filters=[32, 32, 32], 
                                    dropout=0.1, 
                                    image_size=state_dim, 
                                    batch_norm=False, 
                                    layer_norm=True, 
                                    out_emb = hidden_dim)
            
            # Q2
            self.encoder2 = Encoder(inp_channel=C, 
                                    filters=[32, 32, 32], 
                                    dropout=0.1, 
                                    image_size=state_dim, 
                                    batch_norm=False, 
                                    layer_norm=True, 
                                    out_emb = hidden_dim)
        else:
            # 1D
            self.encoder1 = nn.Linear(state_dim, hidden_dim)
            self.encoder2 = nn.Linear(state_dim, hidden_dim)
        
        mlp_input = 2*hidden_dim if self.goal else hidden_dim
        self.l1 = nn.Linear(mlp_input+num_actions, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, 1)

        self.l3 = nn.Linear(mlp_input+num_actions, hidden_dim)
        self.l4 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action, goal=None):
        # Q1
        x = self.encoder1(state)
        if self.goal:
            print(goal.shape)
            goal_emb = self.encoder1(goal)
            x = torch.cat([x, goal_emb], -1)
        x = torch.cat([x, action], -1)
        x = F.relu(self.l1(x))
        q1 = self.l2(x)

        x = self.encoder2(state)
        if self.goal:
            goal_emb = self.encoder2(goal)
            x = torch.cat([x, goal_emb], -1)
        x = torch.cat([x, action], -1)
        x = F.relu(self.l3(x))
        q2 = self.l4(x)

        return q1, q2    

    def Q1(self, state, action, goal=None):
        x = self.encoder1(state)
        if self.goal:
            goal_emb = self.encoder1(goal)
            x = torch.cat([x, goal_emb], -1)
        x = torch.cat([x, action], -1)
        x = F.relu(self.l1(x))
        q1 = self.l2(x)
        return q1

# Single encoder TD7
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
        
        self.q01 = nn.Linear(feature_layer, hdim)    
        self.q1 = nn.Linear(2*zs_dim + hdim, hdim)
        self.q2 = nn.Linear(hdim, hdim)
        self.q3 = nn.Linear(hdim, 1)


        self.q02 = nn.Linear(feature_layer, hdim)        
        self.q4 = nn.Linear(2*zs_dim + hdim, hdim)
        self.q5 = nn.Linear(hdim, hdim)
        self.q6 = nn.Linear(hdim, 1)


    def forward(self, state, action, zsa, zs, goal):
        embeddings = torch.cat([zsa, zs], 1)
        
        if self.obs_type == 'pixel':
            state = self.encoder(state)
            if self.goal:
                goal = self.encoder(goal)
            sa = torch.cat([state, goal, action], 1) if self.goal else torch.cat([state, action], 1)
        else:
            sa = torch.cat([state, goal, action], 1) if self.goal else torch.cat([state, action], 1)
        
        q1 = AvgL1Norm(self.q01(sa))    
        q1 = torch.cat([q1, embeddings], 1)
        q1 = self.activ(self.q1(q1))
        q1 = self.activ(self.q2(q1))
        q1 = self.q3(q1)

        q2 = AvgL1Norm(self.q02(sa))
        q2 = torch.cat([q2, embeddings], 1)
        q2 = self.activ(self.q4(q2))
        q2 = self.activ(self.q5(q2))
        q2 = self.q6(q2)
        return torch.cat([q1, q2], 1)

if __name__ == "__main__":
    inp = torch.rand([10, 3, 64, 64])
    acs = torch.rand([10, 22])

    model = QNetwork((3, 64, 64), num_actions=22, hidden_dim=128)
    out = model(inp, acs)
    print(out[0].shape)

    model = ValueNetwork((3, 64, 64), hidden_dim=128)
    out = model(inp)
    print(out.shape)

    model = Critic((3, 64, 64), num_actions=22, hidden_dim=128, goal=True)
    out = model(inp, acs, inp)
    print(out[0].shape)