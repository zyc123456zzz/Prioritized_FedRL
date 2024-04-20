import argparse
from Sharing import *
from collections import OrderedDict
from typing import List, Dict, Tuple
import os
import numpy as np
import flwr as fl

from Config import *
from FL_agent import AbsoluteRewardRecorder
import FL_agent

def seed_everything(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

seed = 42
seed_everything(seed)

# local RL agent
class RLClient(fl.client.NumPyClient):
    def __init__(
        self,
        model: FL_agent.MLP_Q_Net, # model here must be a neural network, so that we can directly get the parameters from it.
        agent_id = 0,
        episode = 0 # 代表游戏中的一回合结束了（输了或者赢了）
    ) -> None:
        self.model = model
        self.agent_id = agent_id
        self.episode = episode
    def get_parameters(self, config) -> List[np.ndarray]:
        # 将PyTorch张量表示的参数值转换为NumPy数组，并且确保在CPU上处理。
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        # 这一部分创建了一个有序字典，其中键是模型参数的名称，
        # 而值是对应的PyTorch张量（由 torch.tensor(v) 创建）。
        # 这确保了参数字典中的值都被转换为PyTorch张量。
        state_dict = OrderedDict({k:torch.tensor(v) for k, v in params_dict})
        # strict=True 表示加载过程将是严格的，
        # 即只有在参数字典中包含的参数名称与模型中的参数名称完全匹配时才会成功加载，否则会抛出异常。
        self.model.load_state_dict(state_dict, strict=True)
    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        # we need to make everything clare before we dive into the training process.
        self.set_parameters(parameters)
        # local agent update is done here.
        # different env must be transferred into this function
        _, reward, delta_episode = FL_agent.double_dqn(env=env_train, net=self.model, **agent_trainer_hyperparameters)
        self.episode += delta_episode
        return self.get_parameters(config={}), reward, {}
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        # evaluation is on the local dataset.
        # Our purpose is to test the efficiency of the policy.
        # Use global RLagent network.
        self.set_parameters(parameters)
        # invoke the inner function to get the reward.
        # Here the net is not on the DEVICE.
        train_recorder_len = train_recorder.record(net=self.model.to(DEVICE))
        validation_recorder_len = validation_recorder.record(net=self.model.to(DEVICE))
        # test_recorder.record(net=self.model)
        # report should be done in server port.
        '''if not silent_flag:
            # Report error
            if (self.episode+1) % show_interval == 0:
                train_recorder.report()
                validation_recorder.report()'''
        train_step = train_recorder.current_step
        # print(f'The current train_step is:{train_step}, and the train cum_reward is:{train_recorder.cum_reward}')
        reward = train_recorder.cum_reward[train_step-1]
        return reward, 1, {}

if __name__ == "__main__":
    """Load data, start CifarClient."""
    # argparser;
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--RLAgent-id", type=int, required=True, choices=range(0, N), help="Your Client ID")
    args = parser.parse_args()
    AgentId=args.RLAgent_id

    # different Agent has different log dir
    log_dir = global_log_dir + f'Agent-ID_{AgentId}/'

    # mkdir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists('./' + save_dir_name):
        os.makedirs('./' + save_dir_name)
    seed4 = 66
    np.random.seed(seed4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed4)

    # Prepare some parameters: most are loaded from Config.py
    net_class = FL_agent.MLP_Q_Net


    # Here we sample multiple training theta.
    train_theta = np.random.uniform(0 ,1,size=N)
    print(f"--------------{AgentId} training theta is {train_theta[AgentId]}-------------")

    validation_theta = np.random.uniform()
    test_theta = np.random.uniform()
    np.save(log_dir + 'train_theta.npy', train_theta)
    np.save(log_dir + 'validation_theta.npy', validation_theta)
    np.save(log_dir + 'test_theta.npy', test_theta)

    train_theta = np.load(log_dir + 'train_theta.npy')
    validation_theta = np.load(log_dir + 'validation_theta.npy')
    test_theta = np.load(log_dir + 'test_theta.npy')

    recorder_class = AbsoluteRewardRecorder

    # prepare for the env
    env_train = env_class(para=train_theta[AgentId])
    # if we use checkpoints, then we can set these parameters.
    continue_train_flag = False
    continue_train_para = None # 如果是从文件中加载出来的网络参数，则需要先传到这里然后再进行训练；

    # recorder path
    P_log_dir = log_dir + 'Prioritized-' + 'm=' + str(merge_interval) + 'out=' + str(0) + '/'
    if not os.path.exists(P_log_dir):
        os.makedirs(P_log_dir)
        os.makedirs(P_log_dir + 'train_P/')
        os.makedirs(P_log_dir + 'validation_P/')
        # os.makedirs(P_log_dir + 'test_P/')
    # Prepare recorder
    train_recorder_log_dir = P_log_dir + 'train_P/'
    validation_recorder_log_dir = P_log_dir + 'validation_P/'
    # test_recorder_log_dir = P_log_dir + 'test_P/'
    train_recorder_name = 'Train recorder PRIORITY'
    validation_recorder_name = 'Validation recorder PRIORITY'
    # test_recorder_name = 'Test recorder PRIORITY'
    record_len = outer_loop_iter_upper_bound  # We simply record every outer loop

    # Load model and data
    # Use RLagent class to continue the training process.
    # get the agent_id and start FL training process.
    # 对于每一个进程来讲就是一个独立的训练个体，所以一般来讲不会出现继续训练的概念，
    # 默认初始开始训练，除非从mem中导入参数进行训练

    obs_size = env_train.observation_space.shape[0]
    n_actions = env_train.action_space.n # action_space.n
    model = FL_agent.MLP_Q_Net(obs_size, n_actions, **net_extra_hyperparameters)
    # 定义一个个体训练的方法， 需要对当前client提供的net进行训练，所以需要传入net进行训练
    if continue_train_flag:
        # The net has existing parameters.
        model.load_state_dict(continue_train_para)
    # For different client, we should firstly initialize the recorder class
    # because Flower framework does not support the parameters transmission of recorder class
        # Initialize a recorder object.
    # print(f"Agent :{agent_id}'s train theta is: {train_theta}")
    train_recorder = recorder_class(env_class=env_class,
                                    theta=train_theta[AgentId],
                                    log_folder_dir=train_recorder_log_dir,
                                    record_len=record_len,
                                    recorder_name=train_recorder_name,
                                    gamma=gamma,
                                    **recorder_extra_hyperparameters)

    validation_recorder = recorder_class(env_class=env_class,
                                         theta=validation_theta,
                                         log_folder_dir=validation_recorder_log_dir,
                                         record_len=record_len,
                                         recorder_name=validation_recorder_name,
                                         gamma=gamma,
                                         **recorder_extra_hyperparameters)
    '''test_recorder = recorder_class(env_class=env_class,
                                    theta=test_theta,
                                    log_folder_dir=test_recorder_log_dir,
                                    record_len=record_len,
                                    recorder_name=test_recorder_name,
                                    gamma=gamma,
                                    **recorder_extra_hyperparameters)'''
    print(f"---------------------- {AgentId} START TRAINING! ---------------------------")
    client = RLClient(model, agent_id=AgentId)
    fl.client.start_client(server_address="127.0.0.1:8080", client=client.to_client())
    print(f"----------------{AgentId} is on {client.episode}!------------------")