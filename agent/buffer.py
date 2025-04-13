import numpy as np 
import torch 
import random

class ReplayBuffer(object):
    def __init__(self, capacity):

        self.capacity = capacity
        self.buffer = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ptr = 0
        self.size = 0
    
    def add(self, 
            observation, 
            action, 
            next_observation, 
            reward, 
            done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.ptr] = (observation, action, reward, next_observation, done)
        self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def add_batch(self, 
                  observations,
                  actions,
                  next_observations,
                  rewards,
                  dones):
        
        if len(dones.shape) > 1:
            B, _ = dones.shape
        for i in range(B):
            self.add(observations[i], actions[i], rewards[i], next_observations[i], dones[i])
    
    def __len__(self):
        return len(self.buffer)

class LAP(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        max_size=1e6,
        batch_size=256,
        max_action=1,
        normalize_actions=True,
        prioritized=True,
        obs_type=None,
        offline=False
    ):
    
        max_size = int(max_size)
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.device = device
        self.batch_size = batch_size
        self.obs_type = obs_type
        if not offline:
            self.state = np.zeros((max_size, *state_dim)) if obs_type == 'pixel' else np.zeros((max_size, state_dim))
            self.action = np.zeros((max_size, action_dim))
            self.next_state = np.zeros((max_size, *state_dim)) if obs_type == 'pixel' else np.zeros((max_size, state_dim))
            self.reward = np.zeros((max_size, 1))
            self.not_done = np.zeros((max_size, 1))

        self.prioritized = prioritized
        if prioritized:
            self.priority = torch.zeros(max_size, device=device)
            self.max_priority = 1

        self.normalize_actions = max_action if normalize_actions else 1

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action/self.normalize_actions
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        
        if self.prioritized:
            self.priority[self.ptr] = self.max_priority

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        if self.prioritized:
            csum = torch.cumsum(self.priority[:self.size], 0)
            val = torch.rand(size=(self.batch_size,), device=self.device)*csum[-1]
            self.ind = torch.searchsorted(csum, val).cpu().data.numpy()
        else:
            self.ind = np.random.randint(0, self.size, size=self.batch_size)

        return (
            torch.tensor(self.state[self.ind], dtype=torch.float, device=self.device),
            torch.tensor(self.action[self.ind], dtype=torch.float, device=self.device),
            torch.tensor(self.next_state[self.ind], dtype=torch.float, device=self.device),
            torch.tensor(self.reward[self.ind], dtype=torch.float, device=self.device),
            torch.tensor(self.not_done[self.ind], dtype=torch.float, device=self.device)
        )


    def update_priority(self, priority):
        self.priority[self.ind] = priority.reshape(-1).detach()
        self.max_priority = max(float(priority.max()), self.max_priority)


    def reset_max_priority(self):
        if self.prioritized:
            self.max_priority = float(self.priority[:self.size].max())

class OfflineBuffer(LAP):
    """
    Specifically for Offline Data with goal conditioning 
    """
    def __init__(self, *args, **kwargs):
        super(OfflineBuffer, self).__init__(*args, **kwargs)
    
    def load_ogbench(self, dataset_path, weight=None):
        self.dataset = 'ogbench'
        self.weight = weight
        from .ogb_utils import load_dataset
        if "visual" in dataset_path:
            obs_type = np.uint8
            self.normalize = 255.0
        else:
            obs_type = np.float32
            self.normalize = 1.0
        dataset = load_dataset(dataset_path, obs_type)

        self.state = dataset['observations']
        self.action = dataset['actions']
        self.next_state = dataset['next_observations']
        # self.reward = None
        # self.not_done = 1 - dataset['terminals']
        self.size = np.prod(dataset['terminals'].shape)
        self.num_traj, self.traj_length = dataset['terminals'].shape 
        if self.prioritized:
            self.priority = torch.ones(self.size).to(self.device)
    
    # def save_trajectory(self):
    #     import imageio, cv2
    #     from similarity import hist
    #     from tqdm import tqdm
    #     from numpy.linalg import norm
    #     frames = []
    #     for i in tqdm(range(len(self.state[0]))):
    #         score = hist(self.state[1][i], self.state[1][-1])
    #         # print(text)
    #         # score2 = np.dot(self.state[0][i].ravel(), self.state[0][-1].ravel())/(norm(self.state[0][i].ravel())*norm(self.state[0][-1].ravel()))
    #         text = str(score)# + '|' +str(score2)
    #         frame = np.concatenate((self.state[1][i], self.state[1][-1]), axis=0)
    #         frame = cv2.putText(frame, text, (0,64), cv2.FONT_HERSHEY_COMPLEX_SMALL, 
    #                0.5, (0,255, 255), 1, 1)
    #         frames.append(frame)
    #     imageio.mimsave(f'episode_new.mp4', frames, fps=10)

    def sample(self, batch_size, gc_negative=False):
        idxs = np.random.randint(0, self.num_traj, batch_size)
        traj_pos = np.random.randint(0, self.traj_length-1, batch_size)
        # traj_pos to end_pos 
        # distances = np.random.rand(batch_size)  # in [0, 1)

        # distances = np.where(distances < 0.20, 0, distances)
        # weight = 1 - distances
        # weight = np.where(weight > 0.1, weight, 0)
        
        # pos = ((self.traj_length-1 - np.minimum(traj_pos, self.traj_length-1)) * distances).astype(int)
        # np.minimum(p_idxs + offsets, final_state_idxs)
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

        batch = {}
        batch['observation'] = torch.tensor(self.state[idxs, traj_pos]/self.normalize, dtype=torch.float, device=self.device)
        batch['next_observation'] = torch.tensor(self.next_state[idxs, traj_pos]/self.normalize, dtype=torch.float, device=self.device)
        batch['action'] = torch.tensor(self.action[idxs, traj_pos], dtype=torch.float, device=self.device)
        batch['goal'] = torch.tensor(self.state[idxs, goal_pos], dtype=torch.float, device=self.device)
        success = success.astype(float)
        batch['not_done'] = torch.tensor(1 - success, dtype=torch.float, device=self.device)
        reward = success - (1.0 if gc_negative else 0.0)
        batch['reward'] = torch.tensor(reward, dtype=torch.float, device=self.device)
        batch['weight'] = torch.tensor(weight, dtype=torch.float, device=self.device)
        # batch['active'] = active
        # TODO Augment
        return (
            batch['observation'],
            batch['action'],
            batch['next_observation'],
            batch['reward'].unsqueeze(1),
            batch['not_done'].unsqueeze(1),
            batch['goal'],
            batch['weight'].unsqueeze(1)
        )

    def sampleOGB(self, batch_size, gc_negative=True):
        # sample batch size of trajectory
        idxs = np.random.randint(0, self.num_traj, batch_size)
        traj_pos = np.random.randint(0, self.traj_length, batch_size)
        # sample_goal 
        idxs_p, goal_p, active = self.sample_goals(idxs, traj_pos)
        # print( (np.logical_and((idxs == idxs_p), (traj_pos == goal_p)) == active).sum() == batch_size)
        batch = {}
        batch['observation'] = torch.tensor(self.state[idxs, traj_pos]/self.normalize, dtype=torch.float, device=self.device)
        batch['next_observation'] = torch.tensor(self.next_state[idxs, traj_pos]/self.normalize, dtype=torch.float, device=self.device)
        batch['action'] = torch.tensor(self.action[idxs, traj_pos], dtype=torch.float, device=self.device)
        batch['goal'] = torch.tensor(self.state[idxs_p, goal_p], dtype=torch.float, device=self.device)
        success = active.astype(float)
        batch['not_done'] = torch.tensor(1 - success, dtype=torch.float, device=self.device)
        reward = success - (1.0 if gc_negative else 0.0)
        batch['reward'] = torch.tensor(reward, dtype=torch.float, device=self.device)
        # batch['active'] = active
        # TODO Augment
        return (
            batch['observation'],
            batch['action'],
            batch['next_observation'],
            batch['reward'].unsqueeze(1),
            batch['not_done'].unsqueeze(1),
            batch['goal']
        )

    def sample_goals(self, idxs, p_idxs, p_curgoal=0.5, p_trajgoal=0.3, geom_sample=False):
        # random goal 
        # goals _ geometric or uniform normal
        batch_size = len(idxs)
        final_state_idxs = np.array([999]* batch_size)

        if geom_sample:
            # Geometric sampling.
            offsets = np.random.geometric(p=1 - self.discount, size=batch_size)  # in [1, inf)
            middle_goal_idxs = np.minimum(p_idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            middle_goal_idxs = np.round(
                (np.minimum(p_idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            ).astype(int)
        active_2 = np.random.rand(batch_size) < p_curgoal # Current_pos_goal
        active_1 = np.logical_or(active_2, np.random.rand(batch_size) < (p_trajgoal / (1.0 - p_curgoal + 1e-6)))
        idxs = idxs + np.where(active_1, 0, -1)
        goal_pos = middle_goal_idxs
        goal_pos = np.where(active_2, p_idxs, goal_pos) # p_idxs+1
        return idxs, goal_pos, active_2

# if __name__ == "__main__":
#     buffer = OfflineBuffer(
#         39,
#         8,
#         "cpu",
#         max_size=1e3,
#         batch_size=256,
#         max_action=1,
#         normalize_actions=False,
#         prioritized=False,
#     )
#     buffer.load_ogbench("/home/wtc/sai/genrl/data/ogbench/visual-antmaze-medium-navigate-v0-val.npz")
#     batch = buffer.sample(50)
#     print((batch['goal'][batch['active']] - batch['observation'][batch['active']]).sum())