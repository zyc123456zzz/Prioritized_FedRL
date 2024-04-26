from Config import *
from MyCartPole import MyCartPoleEnv
from MyAcrobot import MyAcrobotEnv
from MyMountainCar import MountainCarEnv

train_episodes_upper_bound = 18 # 768 18 1000
merge_interval = 2 # 48 2 40
meta_merge = 1     # 12 1 10
assert train_episodes_upper_bound % merge_interval == 0  # This assumption is for convenience
outer_loop_iter_upper_bound = int(train_episodes_upper_bound / merge_interval)
record_len = outer_loop_iter_upper_bound  # We simply record every outer loop

# First apply single merge_interval to this problem setting.
'''
save_dir_name = 'plot-E_DQNAvg-CartPole-Distributed'
global_log_dir = './E_DQNAvg-CartPole-Distributed/'
'''
'''
save_dir_name = 'plot-P_DQNAvg-CartPole-Distributed'
global_log_dir = './P_DQNAvg-CartPole-Distributed/' 
'''
'''
save_dir_name = 'plot-R_DQNAvg-CartPole-Distributed'
global_log_dir = './R_DQNAvg-CartPole-Distributed/'
'''
'''
save_dir_name = 'plot-DQNAvg-CartPole-Distributed'
global_log_dir = './DQNAvg-CartPole-Distributed/'
'''
'''
save_dir_name = 'plot-P_Momentum_DQNAvg-CartPole-Distributed'
global_log_dir = './P_Momentum_DQNAvg-CartPole-Distributed/' 
'''


'''
save_dir_name = 'plot-E_DQNAvg-Acrobot-Distributed'
global_log_dir = './E_DQNAvg-Acrobot-Distributed/'
'''

save_dir_name = 'plot-P_DQNAvg-Acrobot-Distributed'
global_log_dir = './P_DQNAvg-Acrobot-Distributed/'

'''
save_dir_name = 'plot-R_DQNAvg-Acrobot-Distributed'
global_log_dir = './R_DQNAvg-Acrobot-Distributed/'
'''
'''
save_dir_name = 'plot-DQNAvg-Acrobot-Distributed'
global_log_dir = './DQNAvg-Acrobot-Distributed/'
'''
'''
save_dir_name = 'plot-P_Momentum_DQNAvg-Acrobot-Distributed'
global_log_dir = './P_Momentum_DQNAvg-Acrobot-Distributed/'
'''


'''
save_dir_name = 'plot-E_DQNAvg-MountainCar-Distributed'
global_log_dir = './E_DQNAvg-MountainCar-Distributed/'
'''
'''
save_dir_name = 'plot-P_DQNAvg-MountainCar-Distributed'
global_log_dir = './P_DQNAvg-MountainCar-Distributed/'
'''
'''
save_dir_name = 'plot-R_DQNAvg-MountainCar-Distributed'
global_log_dir = './R_DQNAvg-MountainCar-Distributed/'
'''
'''
save_dir_name = 'plot-DQNAvg-MountainCar-Distributed'
global_log_dir = './DQNAvg-MountainCar-Distributed/'
'''
'''
save_dir_name = 'plot-P_Momentum_DQNAvg-MountainCar-Distributed'
global_log_dir = './P_Momentum_DQNAvg-MountainCar-Distributed/'
'''

silent_flag = SILIENT_FLAG

# Init Hyperparameters
init_epsilon = INIT_EPSILON  # While training in QAvg, init_epsilon will also decay
epsilon_decay = EPS_DECAY  # This is epsilon decay in double DQN, not in QAvg
tgt_net_sync = TGT_NET_SYNC
batch_size = BATCH_SIZE
hidden_size = HIDDEN_SIZE  # Hidden size for one-layer MLP DQN network: MLP_Q_Net
net_extra_hyperparameters = {'hidden_size': hidden_size}
evaluate_episodes_for_conv = EVALUATE_EPISODES_FOR_CONV
solve_criterion = SOLVE_CRITERION
replay_size = REPLAY_SIZE
gamma = GAMMA
lr = 1e-4 # LR Cartpole MountainCar 1e-3   Acrobot 1e-4
# train_episodes_upper_bound should be specified later on
agent_trainer_hyperparameters = {'epsilon': init_epsilon,
                                    'epsilon_decay': epsilon_decay,
                                    'sync_interval': tgt_net_sync,
                                    'gamma': gamma,
                                    'batch_size': batch_size,
                                    'lr': lr,
                                    'evaluate_episodes_for_conv': evaluate_episodes_for_conv,
                                 'train_episodes_upper_bound': merge_interval,
                                    'solve_criterion': solve_criterion,
                                    'replay_size': replay_size,
                                    'silent_flag': silent_flag}
show_interval = SHOW_INTERVAL
evaluate_episodes_for_eval = EVALUATE_EPISODES_FOR_EVAL
recorder_extra_hyperparameters = {'evaluate_episodes_for_eval': evaluate_episodes_for_eval}

# env_class = MyCartPoleEnv
env_class = MyAcrobotEnv
# env_class = MountainCarEnv

SAY=True
N = 50 # control the client number.
S = 5 # control the outer loop number