import argparse
import torch
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.multi_agent_env import MultiAgentEnv
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.attention_sac import AttentionSAC
import gym

gym.logger.set_level(40)

def make_parallel_env(env_id, n_rollout_threads, seed, num_controlled_lagents, num_controlled_ragents, reward_type, render):
    def get_env_fn(rank):
        def init_env():
            env = MultiAgentEnv(env_id, num_controlled_lagents, num_controlled_ragents, reward_type, render)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def run(config):
    USE_CUDA = False
    if config.gpu:
        if torch.cuda.is_available():
            USE_CUDA = True
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) + 1
    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))
    
#     model_run = 'run%i' % max(exst_run_nums)
#     model_path = model_dir / model_run / 'model.pt'

    torch.manual_seed(run_num)
    np.random.seed(run_num)
    env = make_parallel_env(config.env_id, config.n_rollout_threads, run_num,
                            config.n_controlled_lagents, config.n_controlled_ragents, config.reward_type, config.render)
    model = AttentionSAC.init_from_env(env,
                                       tau=config.tau,
                                       pi_lr=config.pi_lr,
                                       q_lr=config.q_lr,
                                       gamma=config.gamma,
                                       pol_hidden_dim=config.pol_hidden_dim,
                                       critic_hidden_dim=config.critic_hidden_dim,
                                       attend_heads=config.attend_heads,
                                       reward_scale=config.reward_scale)
    
#     model = AttentionSAC.init_from_save_(model_path, load_critic=False, gpu=USE_CUDA)
    
    replay_buffer = ReplayBuffer(config.buffer_length, model.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])
    best_rewards = 0
    t = 0
    num_episodes = 0
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        
        if ep_i % (config.epoch_size * config.n_rollout_threads) == 0:
            stat = dict()
            stat['epoch'] = int(ep_i / (config.epoch_size * config.n_rollout_threads) + 1)
            
        obs = env.reset()
        model.prep_rollouts(device='cpu')
        
        s = dict()
        s['dones'] = [0 for i in range(config.n_rollout_threads)]
        s['num_episodes'] = [0 for i in range(config.n_rollout_threads)]
        s['reward'] = [0 for i in range(config.n_rollout_threads)]
        s['success'] = [0 for i in range(config.n_rollout_threads)]
        s['steps_taken'] = [0 for i in range(config.n_rollout_threads)]
        s['reward_buffer'] = [0 for i in range(config.n_rollout_threads)]
        s['steps_buffer'] = [0 for i in range(config.n_rollout_threads)]

        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(model.nagents)]
            # get actions as torch Variables
            torch_agent_actions = model.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, infos = env.step(actions)
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if USE_CUDA:
                    model.prep_training(device='gpu')
                else:
                    model.prep_training(device='cpu')
                for u_i in range(config.num_updates):
                    sample = replay_buffer.sample(config.batch_size,
                                                  to_gpu=USE_CUDA)
                    model.update_critic(sample, logger=logger)
                    model.update_policies(sample, logger=logger)
                    model.update_all_targets()
                model.prep_rollouts(device='cpu')
                
            for i in range(config.n_rollout_threads):
                s['reward'][i] += np.mean(rewards[i])
                s['steps_taken'][i] += 1
                if dones[i][0] == True:
                    s['dones'][i] += 1
                    s['num_episodes'][i] += 1
                    s['reward_buffer'][i] = s['reward'][i]
                    s['steps_buffer'][i] = s['steps_taken'][i]
                    if infos[i]['score_reward'] == 1:
                        s['success'][i] += 1
                if et_i == config.episode_length-1:
                    if dones[i][0] == False:
                        if s['dones'][i] > 0:
                            s['reward'][i] = s['reward_buffer'][i]
                            s['steps_taken'][i] = s['steps_buffer'][i]
                        else:
                            s['num_episodes'][i] += 1
                            
        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)
        global_ep_rews = 0
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalars('agent%i/rewards' % a_i, {'mean_episode_rewards': a_ep_rew}, ep_i)
            global_ep_rews += a_ep_rew / (config.n_controlled_lagents + config.n_controlled_ragents)
        logger.add_scalars('global', {'global_rewards': global_ep_rews}, ep_i)
        
        if global_ep_rews > 0.007:
            model.save(run_dir / ('model_ep%i.pt' % ep_i))
#             print('model saved at ep%i' % ep_i)   
#             print('saved model reward: ', global_ep_rews)
        
        if global_ep_rews > best_rewards:
            best_rewards = global_ep_rews
            if best_rewards > 0.005:
                model.save(run_dir / ('best_model_ep%i.pt' % ep_i))
#                 print('best model saved at ep%i' % ep_i)
#                 print('best global reward: ', best_rewards)
                
#         if ep_i%500 == 0:
#             print('episode: ', ep_i)
#             print('global reward: ', global_ep_rews)
#             print('best global reward: ', best_rewards)

        if ep_i % config.save_interval < config.n_rollout_threads:
            model.prep_rollouts(device='cpu')
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            model.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            model.save(run_dir / 'model.pt')
            
        # An exact episode means a real episode in the game, rather than the episode in a training loop
        # Mean (exact) episode data are only generated from complete exact episodes
        # We calculate the mean (exact) episode data in each epoch
        # (config.epoch_size * config.n_rollout_threads) means the number of training episodes an epoch includes
        # The mean (exact) episode data are used for visualization and comparison
        # Reward, Steps-Taken, Success

        stat['num_episodes'] = stat.get('num_episodes', 0) + np.sum(s['num_episodes'])
        stat['reward'] = stat.get('reward', 0) + np.sum(s['reward'])
        stat['success'] = stat.get('success', 0) + np.sum(s['success'])
        stat['steps_taken'] = stat.get('steps_taken', 0) + np.sum(s['steps_taken'])

        if (ep_i+config.n_rollout_threads) % (config.epoch_size * config.n_rollout_threads) == 0:
            num_episodes += stat['num_episodes']
            print('Epoch {}'.format(stat['epoch']))
            print('Episode: {}'.format(num_episodes))
            print('Reward: {}'.format(stat['reward']/stat['num_episodes']))
            print('Success: {:.2f}'.format(stat['success']/stat['num_episodes']))
            print('Steps-Taken: {:.2f}'.format(stat['steps_taken']/stat['num_episodes']))

    model.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_controlled_lagents", default=2, type=int)
    parser.add_argument("--n_controlled_ragents", default=0, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--epoch_size", default=50, type=int)
    parser.add_argument("--n_episodes", default=50000, type=int)
    parser.add_argument("--episode_length", default=200, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--num_updates", default=4, type=int,
                        help="Number of updates per update cycle")
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for training")
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--pol_hidden_dim", default=128, type=int)
    parser.add_argument("--critic_hidden_dim", default=128, type=int)
    parser.add_argument("--attend_heads", default=4, type=int)
    parser.add_argument("--pi_lr", default=0.001, type=float)
    parser.add_argument("--q_lr", default=0.001, type=float)
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--reward_scale", default=100., type=float)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--gpu", action='store_true')
    parser.add_argument("--reward_type",
                        default="scoring", type=str,
                        choices=['scoring', 'checkpoints'])

    config = parser.parse_args()

    run(config)
