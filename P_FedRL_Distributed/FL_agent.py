import torch.nn as nn
from typing import Type
import torch.nn.functional as F
import os
import ptan
import gym
import numpy as np
import torch.optim as optim
from MyEnv import MyEnv
from MyCartPole import MyCartPoleEnv
from Config import *
from functools import wraps




'''os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10088'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10088'''

# basic training network;
class RLNet(nn.Module):
    def __init__(self, obs_size, n_actions=-1, action_size=-1, **kwargs):
        super(RLNet, self).__init__()
        self.action_size = action_size
        self.obs_size = obs_size
        self.n_actions = n_actions

class MLP_Q_Net(RLNet):
    def __init__(self, obs_size, n_actions, hidden_size):
        super(MLP_Q_Net, self).__init__(obs_size, n_actions)
        self.net = nn.Sequential(
            nn.Linear(self.obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.n_actions)
        )
    def forward(self, x):
        return self.net(x.float())

@torch.no_grad()
def unpack_batch(batch, net, gamma):
    # net is target net, but if we use ddqn, action is selected by previous Q-network.
    # 计算下一状态的Q value function的网络是target network
    states = []
    actions = []
    rewards = []
    done_masks = []
    # 可能是st+1？？？
    last_states = []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        done_masks.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)

    states_v = torch.tensor(states).to(DEVICE)
    actions_v = torch.tensor(actions).to(DEVICE)
    rewards_v = torch.tensor(rewards).to(DEVICE)
    last_states_v = torch.tensor(last_states,).to(DEVICE)

    last_state_q_v = net(last_states_v)
    best_last_q_v = torch.max(last_state_q_v, dim=1)[0] # 这里得到的是在St+1中使得Q最大的那个，但并不是通过argmax得到action后得到的St+1,action下的reward。
    best_last_q_v[done_masks] = 0.0
    return states_v, actions_v, (best_last_q_v * gamma + rewards_v).to(DEVICE)

@torch.no_grad()
def evaluation(network_type,
               net: RLNet,
               env: MyEnv,
               evaluate_episodes_for_eval=10,
               gamma=GAMMA,
               replay_size=REPLAY_SIZE,
               threshold=2.0):
    if network_type == 'Q':
        # for Q-neywork
        selector = ptan.actions.ArgmaxActionSelector()
        agent = ptan.agent.DQNAgent(net, selector, device=DEVICE)
        exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=gamma)
        buffer = ptan.experience.ExperienceReplayBuffer(exp_source,buffer_size=replay_size)

        step = 0
        episode = 0
        evaluate_reward = 0

        while True:
            step += 1
            buffer.populate(1)

            for reward, steps in exp_source.pop_rewards_steps():
                episode += 1
                evaluate_reward += reward
            if episode >= evaluate_episodes_for_eval:
                break

        # return rewards got within evaluate_episodes_for_eval episodes.
        evaluate_reward /= evaluate_episodes_for_eval
        return evaluate_reward, step
    '''if network_type == 'DP':
        # We do not implement this method for policy network
        evaluate_rewards = 0.0
        for _ in range(evaluate_episodes_for_eval):
            obs = env.reset()
            steps = 0
            # 走evaluate_episodes_for_eval步来估计当前的价值
            while True:
                obs_v = ptan.agent.float32_preprocessor([obs]).to(DEVICE)
                mu_v = self(obs_v)
                action = mu_v.squeeze(dim=0).data.cpu().numpy()
                action = np.clip(action, -threshold, threshold)
                obs, reward, done, _ = env.step(action)
                evaluate_rewards += reward
                steps += 1
                if steps >= 200:
                    break
                if done:
                    break
        # averaged reward of evaluate_episodes_for_eval steps reward gotten by agent
        evaluate_rewards /= evaluate_episodes_for_eval
        return evaluate_rewards'''
    return

def double_dqn(env: MyEnv,
               net: MLP_Q_Net,
               epsilon=INIT_EPSILON,  #
               epsilon_decay=EPS_DECAY,  #
               sync_interval=TGT_NET_SYNC,  #
               gamma=GAMMA,  #
               batch_size=BATCH_SIZE,
               lr=LR,
               evaluate_episodes_for_conv=EVALUATE_EPISODES_FOR_CONV,
               solve_criterion=SOLVE_CRITERION,
               replay_size=REPLAY_SIZE,
               train_episodes_upper_bound=TRAIN_EPISODES_UB,  ## merge_interval : 因为这里是每个智能体的训练过程，每次就把当前的训练回合结束就好。
               silent_flag=True,
               evaluate_episodes_for_eval=EVALUATE_EPISODES_FOR_EVAL
               # episode = 0 # why do we need this parameter? we must know
               ):
    print("-----------------START TRAINING!-----------------")
    net = net.to(DEVICE)
    tgt_net = ptan.agent.TargetNet(net) # 是否需要传到device上

    selector = ptan.actions.ArgmaxActionSelector()
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=epsilon, selector=selector)
    agent = ptan.agent.DQNAgent(net, selector, device=DEVICE)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=gamma)
    replay_buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=replay_size)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    step = 0
    # episode = 0 # 需要设置成可以传递的变量，不能每次进这个函数都要被初始化；
    evaluate_reward = 0
    solved = False
    episode = 0

    # silent_flag means whether do we want to print the string or not.
    while True:
        step+=1
        # print("The current step is: "+ str(step))

        replay_buffer.populate(1)

        for reward, steps in exp_source.pop_rewards_steps(): # episode执行完后才会返回该episode积累的经验。
            episode +=1
            # print("The current episode is: " + str(episode))
            evaluate_reward += reward
            if not silent_flag:
                print("%d: episode %d done, reward=%.3f, epsilon=%.2f" % (
                    step, episode, reward, selector.epsilon))
            if episode % evaluate_episodes_for_conv == 0:
                evaluate_reward /= evaluate_episodes_for_conv
                # print("evaluate_reward is " + str(evaluate_reward))
                solved = evaluate_reward > solve_criterion
                evaluate_reward = 0
        if solved:
            if not silent_flag:
                print("Congrats!")
            # print(f"Solved Break:{solved}")
            break
        if episode >= train_episodes_upper_bound:
            if not silent_flag:
                print("Exceed episode upper bound!")
            # print(f"UPPER BREAK:{episode},{train_episodes_upper_bound}")
            break
        if len(replay_buffer) < 2 * batch_size:
            # 看看是否能填满这个replay_buffer
            # print("The current replay_buffer is: " + str(len(replay_buffer)))
            continue

        batch = replay_buffer.sample(batch_size)
        states_v, actions_v, tgt_q_v = unpack_batch(batch, tgt_net.target_model, gamma)
        optimizer.zero_grad()
        q_v = net(states_v)
        q_v = q_v.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        loss_v = F.mse_loss(q_v, tgt_q_v)
        loss_v.backward()
        optimizer.step()
        selector.epsilon *= epsilon_decay

        if episode % sync_interval == 0:
            tgt_net.sync()
    # evaluate after fit
    '''reward, _ = evaluation('Q',
                           net=net,
                           env=env,
                           evaluate_episodes_for_eval=evaluate_episodes_for_eval, # 5
                           gamma=gamma)'''
    # print("-----------------TRAINING PART IS DEAD!-----------------")
    return net, 1, episode # int(reward)


def record_wrapper(record_method):
    @wraps(record_method)
    def wrapRecordMethod(self, **kwargs):
        if self.done:
            print('The recorder is full, do not call record any more.')
            return
        # Call record
        record_method(self, **kwargs)
        # Update self.current_step and self.done
        # print("record_wrapper :" + str(self.current_step))
        self.current_step += 1

        if self.current_step == self.record_len:
            self.done = True
            self.write_log()
        return self.current_step
    return wrapRecordMethod

# Wrapper for class method: report
def report_wrapper(report_method):
    @wraps(report_method)
    def wrapReportMethod(self, **kwargs):
        if self.current_step == 0:
            print('The recorder is empty, please call report later.')
            return
        print(CUTTING_LINE)
        # They have different recorder name actually.
        print(self.recorder_name + ': Current step is ' + str(self.current_step) + '/' + str(self.record_len))
        # Call report
        report_method(self, **kwargs)
        return
    return wrapReportMethod

# Wrapper for class method: get full report
def get_full_report_wrapper(gfr_method):
    @wraps(gfr_method)
    def wrapGFRMethod(self):
        if not self.done:
            print('Warning: the recorder is not full, and you call get_full_report.')
        # Call and return class method
        return gfr_method(self)
    return wrapGFRMethod
class ExperimentRecorder:
    def __init__(self,
                 env_class: Type[MyEnv],
                 theta: float, # record different theta training process.
                 log_folder_dir: str,
                 record_len: int,
                 recorder_name: str,
                 gamma: float,
                 **kwargs):
        self.recorder_type_list = None   # This is just for avoid warning
        self.env_class = env_class
        self.theta = theta
        self.log_folder_dir = log_folder_dir
        self.record_len = record_len
        self.recorder_name = recorder_name
        self.gamma = gamma
        self.env_size = 1

        self.current_step = 0
        self.done = False

    @record_wrapper
    def record(self, net: RLNet, **kwargs):
        # Do nothing
        return

    @report_wrapper
    def report(self):
        # Do nothing
        return

    # This method is silent, and usually return an ndarray
    @get_full_report_wrapper
    def get_full_report(self):
        # Do nothing, but must return a list
        return []

    # This method is called only in record method, do not call it by yourself
    # We do not use private method here for convenience, but do not call it by yourself
    def write_log(self):
        # Do nothing
        return

class AbsoluteRewardRecorder(ExperimentRecorder):
    def __init__(self,
                 env_class: Type[MyEnv],
                 theta: float,
                 log_folder_dir: str,
                 record_len: int,
                 recorder_name: str,
                 gamma: float,
                 evaluate_episodes_for_eval: int,
                 **kwargs):
        super(AbsoluteRewardRecorder, self).__init__(env_class=env_class,
                                                     theta=theta,
                                                     log_folder_dir=log_folder_dir,
                                                     record_len=record_len,
                                                     recorder_name=recorder_name,
                                                     gamma=gamma)
        self.evaluate_episodes_for_eval = evaluate_episodes_for_eval
        self.cum_reward = np.zeros(self.record_len)
        # print("Initing AbsoluteRewardRecorder!")
        # self.cum_reward_avg = np.zeros(self.record_len)

    @record_wrapper
    def record(self, net: RLNet, **kwargs):
        # print("Recorder theta is: " + str(self.theta))
        tmp_env = self.env_class(para=self.theta)
        # current step is conducted by report_wrapper.
        # print("recorder: "+ str(self.current_step))
        self.cum_reward[self.current_step], recorder_len = \
                evaluation('Q',
                           net=net,
                           env=tmp_env,
                           evaluate_episodes_for_eval=self.evaluate_episodes_for_eval, # 5
                           gamma=self.gamma)
        # self.cum_reward_avg[self.current_step] = np.average(self.cum_reward[self.current_step])
        return

    @report_wrapper
    def report(self):
        print('Most recent average cumulative reward: ' + str(self.cum_reward_avg[self.current_step-1]))
        return

    @get_full_report_wrapper
    def get_full_report(self):
        if not self.done:
            print('Warning: the recorder is not full, and you call get_full_report.')
        return self.cum_reward

    def write_log(self):
        print("wirte the log file!")
        np.save(self.log_folder_dir + 'cum_reward.npy', self.cum_reward)
        # np.save(self.log_folder_dir + 'cum_reward_avg.npy', self.cum_reward_avg)
        return



if __name__ == "__main__":
    print("Fl_agent start!")