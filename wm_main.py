import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

from pathlib import Path
from collections import defaultdict
from tqdm import tqdm 
import hydra
import numpy as np
import torch
import wandb

import tools.utils as utils
from tools.logger import Logger
from tools.replay import OGReplayBuffer, make_replay_loader
import gymnasium as gym
import ogbench 
torch.backends.cudnn.benchmark = True

def make_agent(obs_type, obs_spec, action_spec, num_expl_steps, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)


def make_dreamer_agent(obs_space, action_spec, cur_config, cfg):
    from copy import deepcopy
    cur_config = deepcopy(cur_config)
    if hasattr(cur_config, 'agent'):
        del cur_config.agent
    return hydra.utils.instantiate(cfg, cfg=cur_config, obs_space=obs_space, act_spec=action_spec)

def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# Not needed for world model
def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)

def reshape_video(v, n_cols=None):
    """Helper function to reshape videos."""
    if v.ndim == 4:
        v = v[None,]

    _, t, h, w, c = v.shape

    if n_cols is None:
        # Set n_cols to the square root of the number of videos.
        n_cols = np.ceil(np.sqrt(v.shape[0])).astype(int)
    if v.shape[0] % n_cols != 0:
        len_addition = n_cols - v.shape[0] % n_cols
        v = np.concatenate((v, np.zeros(shape=(len_addition, t, h, w, c))), axis=0)
    n_rows = v.shape[0] // n_cols

    v = np.reshape(v, newshape=(n_rows, n_cols, t, h, w, c))
    v = np.transpose(v, axes=(2, 5, 0, 3, 1, 4))
    v = np.reshape(v, newshape=(t, c, n_rows * h, n_cols * w))

    return v


def get_wandb_video(renders=None, n_cols=None, fps=15):
    from PIL import Image, ImageEnhance
    """Return a Weights & Biases video.

    It takes a list of videos and reshapes them into a single video with the specified number of columns.

    Args:
        renders: List of videos. Each video should be a numpy array of shape (t, h, w, c).
        n_cols: Number of columns for the reshaped video. If None, it is set to the square root of the number of videos.
    """
    # Pad videos to the same length.
    max_length = max([len(render) for render in renders])
    for i, render in enumerate(renders):
        assert render.dtype == np.uint8

        # Decrease brightness of the padded frames.
        final_frame = render[-1]
        final_image = Image.fromarray(final_frame)
        enhancer = ImageEnhance.Brightness(final_image)
        final_image = enhancer.enhance(0.5)
        final_frame = np.array(final_image)

        pad = np.repeat(final_frame[np.newaxis, ...], max_length - len(render), axis=0)
        renders[i] = np.concatenate([render, pad], axis=0)

        # Add borders.
        renders[i] = np.pad(renders[i], ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
    renders = np.array(renders)  # (n, t, h, w, c)

    renders = reshape_video(renders, n_cols)  # (t, c, nr * h, nc * w)
    
    return wandb.Video(renders, fps=fps, format='mp4')

def make_env(env_name, **env_kwargs):
    return ogbench.make_env_and_datasets(env_name, **env_kwargs)

class Workspace:
    def __init__(self, cfg, savedir=None, workdir=None,):
        self.workdir = Path.cwd() if workdir is None else workdir
        print(f'workspace: {self.workdir}')

        self.cfg = cfg
        
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # create logger
        self.logger = Logger(self.workdir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)
        # create envs
        self.task = task = cfg.task
        img_size = cfg.img_size

        
        self.train_env = ogbench.make_env_and_datasets(cfg.task, env_only=True)
        h, w, c = self.train_env.observation_space.shape
        observation_space = dict(observation = gym.spaces.Box(low=0, high=255, shape=(c, h, w), dtype=np.uint8),
                                is_first = gym.spaces.Box(low=0, high=1, shape = (), dtype=bool),
                                is_last = gym.spaces.Box(low=0, high=1, shape = (), dtype=bool),
                                is_terminal = gym.spaces.Box(low=0, high=1, shape = (), dtype=bool),
                                goal = gym.spaces.Box(low=0, high=255, shape=(c, h, w), dtype=np.uint8),
                                )
        action_space = self.train_env.action_space
        data_specs = (
            observation_space,
            dict(action=action_space),
            dict(reward=gym.spaces.Box(low=0, high=1, shape = (1,), dtype=np.float32)),
            dict(discount=gym.spaces.Box(low=0, high=1, shape = (1,), dtype=np.float32)),
        )

        # # create agent 
        sample_agent = make_dreamer_agent(observation_space, action_space, cfg, cfg.agent)

        if cfg.train_from_data:
            # Loading replay buffer
            if cfg.replay_from_wandb_project is not None:
                api = wandb.Api()
                project_name = cfg.replay_from_wandb_project
                params2search = {
                    "task" : cfg.task if cfg.task_snapshot is None else cfg.task_snapshot,
                    "seed" : cfg.seed if cfg.seed_snapshot is None else cfg.seed_snapshot,
                }
                runs = api.runs(f"PUT_YOUR_USER_HERE/{project_name}")
                found = False
                for run in runs:
                    if np.all([ v == run.config.get(k, None) for k,v in params2search.items()]):
                        found = True
                        found_path = Path(run.config['workdir'].replace('/code', ''))
                        break
                if not found:
                    raise Exception("Replay from wandb buffer not found")

                replay_dir = found_path / 'code' / 'buffer'
            else:
                replay_dir = Path(cfg.replay_load_dir)
                # eval_replay_dir = Path(cfg.replay_eval_dir) if cfg.replay_eval_dir else None

            # create data storage
            self.replay_storage = OGReplayBuffer(data_specs, [],
                                                    replay_dir,
                                                    length=cfg.batch_length, **cfg.replay,
                                                    device=cfg.device, ignore_extra_keys=True, load_recursive=True)
            print('Loaded ', self.replay_storage._loaded_episodes, 'episodes from ', str(replay_dir))

            # create replay buffer
            self.replay_loader = make_replay_loader(self.replay_storage,
                                                    cfg.batch_size,)
            self._replay_iter = None

        if cfg.snapshot_load_dir is not None:
            snapshot_dir = Path(cfg.snapshot_load_dir)
        else:
            snapshot_dir = None

        if snapshot_dir is not None:        
            self.load_snapshot_td7(snapshot_dir, resume=True)
            if self.cfg.reset_world_model:
                self.agent.wm = sample_agent.wm 
                # To reset optimization
                from agent import dreamer_utils as common
                self.agent.wm.model_opt = common.Optimizer('model', self.agent.wm.parameters(), **self.agent.wm.cfg.model_opt, use_amp=self.agent.wm._use_amp)
            if self.cfg.reset_connector:
                self.agent.wm.connector = sample_agent.wm.connector
                # To reset optimization
                from agent import dreamer_utils as common
                self.agent.wm.model_opt = common.Optimizer('model', self.agent.wm.parameters(), **self.agent.wm.cfg.model_opt, use_amp=self.agent.wm._use_amp)

            # overwriting cfg
            self.agent.cfg = sample_agent.cfg
            self.agent.wm.cfg = sample_agent.wm.cfg 
            
            if self.cfg.reset_imag_behavior:
                self.agent.instantiate_imag_behavior()
        else:
            self.agent = sample_agent
        
        # if hasattr(self.eval_env, 'eval_mode'):
        #     self.eval_env.eval_mode()
        
        eval_specs = (
            observation_space,
            dict(action=action_space),
            dict(reward=gym.spaces.Box(low=0, high=1, shape = (1,), dtype=np.float32)),
            dict(discount=gym.spaces.Box(low=0, high=1, shape = (1,), dtype=np.float32)),
        )
        eval_replay_dir = Path(cfg.replay_eval_dir) if cfg.replay_eval_dir else None
        self.eval_storage = OGReplayBuffer(eval_specs, {},
                                                # self.workdir / 'eval_buffer',
                                                eval_replay_dir,
                                                length=cfg.batch_length, **cfg.replay,
                                                device=cfg.device, ignore_extra_keys=True,)
        ##################### TODO : LOAD VAL DATA FOR evaluation #######################
        self.eval_replay_loader = make_replay_loader(self.eval_storage,
                                                    cfg.batch_size,)
        self._eval_replay_iter = None
        self.eval_storage._minlen = 1


        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self, eval=False):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    @property
    def eval_replay_iter(self):
        if self._eval_replay_iter is None:
            self._eval_replay_iter = iter(self.eval_replay_loader)
        return self._eval_replay_iter
    

    def eval_imag_behavior(self,):
        self.agent._backup_acting_behavior = self.agent._acting_behavior
        self.agent._acting_behavior = self.agent._imag_behavior
        self.evaluate()
        self.agent._acting_behavior = self.agent._backup_acting_behavior

    def train(self):
        # self.agent._acting_behavior.reset_actor()
        train_until_step = utils.Until(self.cfg.num_train_frames, 1)
        eval_every_step = utils.Every(self.cfg.eval_every_frames, 1)
        should_log_scalars = utils.Every(self.cfg.log_every_frames, 1)
        should_save_model = utils.Every(self.cfg.save_every_frames, 1)
        should_log_visual = utils.Every(self.cfg.visual_every_frames, 1)
        metrics = None
        eval_metrics = None
        while train_until_step(self.global_step):
            # try to evaluate
            # if self.global_step > 0 and self.global_step % 2000 == 0: #  
            #     with torch.no_grad(), utils.eval_mode(self.agent):
            #         batch_data = next(self.eval_replay_iter)
            #         eval_metrics = self.agent.act(batch_data, None, self.global_step, eval_mode=True, state=None, batch=True)
            #         self.logger.log_metrics(eval_metrics, self.global_frame, ty='eval')

            if eval_every_step(self.global_step+1):
                
                if self.cfg.eval_modality == 'task':
                    _ = self.evaluate()
                if self.cfg.eval_modality == 'task_imag':
                    self.eval_imag_behavior()
                if self.cfg.eval_modality == 'from_text':
                    self.logger.log('eval_total_time', self.timer.total_time(), self.global_frame)
                    self.eval_from_text()
                if self.cfg.eval_modality == 'data':
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        batch_data = next(self.eval_replay_iter)
                        eval_metrics = self.agent.act(batch_data, None, self.global_step, eval_mode=True, state=None, batch=True)
                        self.logger.log_metrics(eval_metrics, self.global_frame, ty='eval')
                
            if self.cfg.train_from_data:
                # Sampling data
                batch_data = next(self.replay_iter)
                # Set Active OGB
                if self.cfg.train_world_model:
                    # print(f"Update : {self.global_step}")
                    state, outputs, metrics = self.agent.update_wm(batch_data, self.global_step)
                    # get the goal encoded information
                    # start, metrics = self.agent.update_acting_behavior(state, outputs, metrics, batch_data)
                    # metrics = {}
                    # start, metrics = self.agent.update_acting_behavior(None, None, metrics, batch_data)
                else:
                    with torch.no_grad():
                        outputs, metrics = self.agent.wm.observe_data(batch_data,)
                if self.cfg.train_connector:
                    #OGB Do we need to train this for Dreamer v3
                    _, metrics = self.agent.wm.update_additional_detached_modules(batch_data, outputs, metrics)
            else:
                imag_warmup_steps = self.cfg.imag_warmup_steps
            
            if self.global_step > 0:
                # update the metrics
                if should_log_scalars(self.global_step):
                    if hasattr(self, 'replay_storage'):
                        metrics.update(self.replay_storage.stats)
                    self.logger.log_metrics(metrics, self.global_frame, ty='train')
                if should_log_visual(self.global_step) and self.cfg.train_from_data and hasattr(self.agent, 'report'):
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        videos = self.agent.report(next(self.eval_replay_iter))
                        self.logger.log_visual(videos, self.global_frame)
                if should_log_scalars(self.global_step):
                    elapsed_time, total_time = self.timer.reset()
                    with self.logger.log_and_dump_ctx(self.global_frame, ty='train') as log:
                        log('fps', self.cfg.log_every_frames / elapsed_time)
                        log('step', self.global_step)
                        if 'model_loss' in metrics: 
                            log('episode_reward', metrics['model_loss'].item())
                    
                # save last model
                if should_save_model(self.global_step):
                    self.save_last_model()

            self._global_step += 1
            # == 1000 is to make sure everything is going well since the start
            if (self.global_frame == 1000) or (self.global_frame % self.cfg.snapshot_every_frames == 0):
                self.save_snapshot()

    @utils.retry
    def save_snapshot(self):
        snapshot = self.root_dir / f'snapshot_{self.global_frame}.pt'
        keys_to_save = ['agent', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)


    def setup_wandb(self):
        cfg = self.cfg
        exp_name = '_'.join([
            cfg.experiment, cfg.agent.name, cfg.task, cfg.obs_type,
            str(cfg.seed)
        ])
        wandb.init(project=cfg.project_name, group=cfg.agent.name, name=exp_name)
        flat_cfg = utils.flatten_dict(cfg)
        wandb.config.update(flat_cfg)
        self.wandb_run_id = wandb.run.id

    @utils.retry
    def save_last_model(self):
        snapshot = self.root_dir / 'last_snapshot.pt'
        if snapshot.is_file():
            temp = Path(str(snapshot).replace("last_snapshot.pt", "second_last_snapshot.pt"))
            os.replace(snapshot, temp)
        keys_to_save = ['agent', '_global_step', '_global_episode']
        if self.cfg.use_wandb: 
            keys_to_save.append('wandb_run_id')
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    @utils.retry
    def load_snapshot(self, snapshot_dir, resume=True):
        print('Loading snapshot from: ', str(snapshot_dir))
        try:
            snapshot = snapshot_dir / 'last_snapshot.pt' if resume else snapshot_dir
            with snapshot.open('rb') as f:
                payload = torch.load(f, weights_only=False)
        except:
            snapshot = Path(str(snapshot_dir).replace('last_snapshot', 'second_last_snapshot'))
            with snapshot.open('rb') as f:
                payload = torch.load(f)
        if type(payload) != dict:
            self.agent = payload
            self.agent.requires_grad_(requires_grad=False)
            return
        # self.agent._acting_behavior.reset_actor()
        # self.agent.requires_grad_(requires_grad=False)
        for k,v in payload.items():
            setattr(self, k, v)
            if k == 'wandb_run_id' and resume:
                assert wandb.run is None
                cfg = self.cfg
                exp_name = '_'.join([
                    cfg.experiment, cfg.agent.name, cfg.task, cfg.obs_type,
                    str(cfg.seed)
                ])
                wandb.init(project=cfg.project_name, group=cfg.agent.name, name=exp_name)
        # self.agent._acting_behavior.reset_actor()
        # self.agent.requires_grad_(requires_grad=False)
    
    @utils.retry
    def load_snapshot_td7(self, snapshot_dir, resume=True):
        print('Loading snapshot from: ', str(snapshot_dir))
        try:
            snapshot = snapshot_dir / 'last_snapshot.pt' if resume else snapshot_dir
            with snapshot.open('rb') as f:
                payload = torch.load(f, weights_only=False)
        except:
            snapshot = Path(str(snapshot_dir).replace('last_snapshot', 'second_last_snapshot'))
            with snapshot.open('rb') as f:
                payload = torch.load(f)
        if type(payload) != dict:
            self.agent = payload
            self.agent.requires_grad_(requires_grad=False)
            return
        
        for k,v in payload.items():
            setattr(self, k, v)

    def get_snapshot_dir(self):
        snap_dir = self.cfg.snapshot_dir
        snapshot_dir = self.workdir / Path(snap_dir)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        return snapshot_dir 

    def use_world_model(self):
        
        with torch.no_grad(), utils.eval_mode(self.agent):
            data = next(self.replay_iter)# B X T X {Image, action} --> Z_sa
            report = {}
            data = self.agent.wm.preprocess(data) # image [0, 255] -? [-0.5, 0.5]
            for key in self.agent.wm.heads['decoder'].cnn_keys:
                name = key.replace('/', '_')
            #     zsa = self.agent.wm.Zsa(data, key)
            
            # print(zsa.shape)
                report[f'openl_test'] = self.agent.wm.video_pred(data, key, 5)
                # self.logger.log_visual(report, self.global_frame)
    def custom_wandb(self, exp_name, task,seed):
        cfg = self.cfg
        exp_name = '_'.join([
            exp_name, task,
            str(seed)
        ])
        wandb.init(project=cfg.project_name, group=cfg.agent.name, name=exp_name)
        flat_cfg = utils.flatten_dict(cfg)
        wandb.config.update(flat_cfg)
        self.wandb_run_id = wandb.run.id

    def offline_train(self):
        from td7_v2 import Agent
        from tqdm import tqdm 
        
        seed = 123
        exp_name = 'Dreamer_TD7'
        
        # eval_env.seed(args.seed+100)
        torch.manual_seed(seed)
        np.random.seed(seed)
        # load model 
        # load_path = 'F:/workspace/exp_local/2025.04.06/214307_dreamer/100001'
        max_timesteps = 400000
        env_name = 'visual-cube-single-play-v0'
        self.custom_wandb(exp_name, env_name, seed)
        env = ogbench.make_env_and_datasets(dataset_name=env_name, env_only=True)
        env.action_space.seed(seed)
        # eval_env = ogbench.make_env_and_datasets(dataset_name=env_name, env_only=True)
        agent = Agent(state_dim=(3,64,64), 
                      action_dim=env.action_space.shape[0], 
                      max_action=1, 
                      offline=True,
                      wm=self.agent.wm, 
                      goal_cond=True, 
                      obs_type='pixel')
        # agent.load(load_path)
        agent.replay_buffer.load_ogbench('F:/workspace/sai/data/visual-cube-single-play-v0.npz', weight=None)
        evals = []
        for t in tqdm(range(max_timesteps+1)):
            
            evaluate_ogbench(agent, env, evals, eval_tasks=None, t=t)
            
            log_status, metrics = agent.train()
            if log_status:
                wandb.log(metrics, step=t)
            if t > 0 and t%20000==0:
                print(f'{t} Checkpoint Saved')
                agent.save(self.workdir)

def evaluate_ogbench(agent, env, evals, eval_tasks, t):
    if t == 0 or t % 20000 != 0:
        return 
    renders = []
    eval_metrics, overall_metrics = {}, defaultdict(list)
    video_eps = 5
    task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
    num_tasks = eval_tasks if eval_tasks is not None else len(task_infos)
    for task_id in tqdm(range(1, num_tasks + 1)):
        task_name = task_infos[task_id - 1]['task_name']
        eval_info, cur_renders = evaluate_fn(
            agent=agent,
            env=env,
            task_id=task_id,
            args=None,
            num_eval_episodes=20,
            num_video_episodes=video_eps,
            video_frame_skip=3,
            eval_gaussian=None,
        )
        renders.extend(cur_renders)
        metric_names = ['success']
        eval_metrics.update(
            {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
        )
        for k, v in eval_info.items():
            if k in metric_names:
                overall_metrics[k].append(v)
    for k, v in overall_metrics.items():
        eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)
    
    if video_eps > 0:
        video = get_wandb_video(renders=renders, n_cols=num_tasks)
        eval_metrics['video'] = video
    wandb.log(eval_metrics, step=t)
    
    
    # print(f'--------------------- {t} ---------------------')
    tqdm.write(f"Score {eval_metrics['evaluation/overall_success']}/ 1")
    # print(f'--------------------- {t} ---------------------')
    evals.append(eval_metrics)
    # np.save(f"{args.working_dir}/results.npy", evals)

def evaluate_fn(
    agent,
    env,
    task_id=None,
    args=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_gaussian=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        task_id: Task ID to be passed to the environment.
        args: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    # trajs = []
    stats = defaultdict(list)

    renders = []
    for i in range(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes
        observation, info = env.reset(options=dict(task_id=task_id, render_goal=should_render))
        goal = info.get('goal')
        goal_frame = info.get('goal_rendered')
        done = False
        step = 0
        render = []
        while not done:
            # action = agent(observations=observation, goals=goal, temperature=eval_temperature)
            action = agent.select_action(observation, goal)
            action = np.array(action)
            

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                if goal_frame is not None:
                    render.append(np.concatenate([goal_frame, frame], axis=0))
                else:
                    render.append(frame)

            observation = next_observation
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            # trajs.append(traj)
        else:
            add_to(stats, flatten(info))
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, renders


def start_training(cfg, savedir, workdir):
    from wm_main import Workspace as W
    from dataset import load_dataset
    from tqdm import tqdm 
    
    tqdm.write(f'Dataset : {cfg.task}')
    root_dir = Path.cwd()

    if not os.path.exists(cfg.replay_load_dir):
        _ = load_dataset(dataset_path=cfg.train_data, 
                        directory=cfg.replay_load_dir,
                        ob_dtype=np.uint8)
    if not os.path.exists(cfg.replay_eval_dir):
        _ = load_dataset(dataset_path=cfg.eval_data, 
                        directory=cfg.replay_eval_dir,
                        ob_dtype=np.uint8)

    cfg.workdir = str(root_dir)
    workspace = W(cfg, savedir, workdir)
    workspace.root_dir = root_dir
    snapshot = workspace.root_dir / 'last_snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot(workspace.root_dir)
    if cfg.use_wandb and wandb.run is None:
        # otherwise it was resumed
        workspace.setup_wandb()
    
    
    workspace.train() 

@hydra.main(config_path='.', config_name='wm')
def main(cfg):
    start_training(cfg, None, None)

if __name__ == '__main__':
    main()
