"""Flower server example."""
import copy
from typing import List, Tuple, Optional, Callable, Dict, Union
from collections import OrderedDict

import ptan
import torch
from torch import optim, Tensor
import numpy as np
import flwr as fl
from flwr.common.logger import log
from flwr.common import NDArrays, Scalar, Parameters, MetricsAggregationFn, FitIns, parameters_to_ndarrays, \
    EvaluateIns, FitRes, ndarrays_to_parameters, EvaluateRes
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from flwr.server.strategy.aggregate import aggregate_inplace, aggregate
from Sharing import *
import FL_agent
from FL_agent import unpack_batch, MLP_Q_Net
from MyEnv import MyEnv
import os
import sys
import torch.nn as nn
import torch.nn.functional as F




WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""

def seed_everything(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

args=sys.argv[1] # 从bash脚本中得到当前server的参数。
WARNING = 30
BETA = 1
recorder_log_dir = global_log_dir + 'Server/' + f'{args}/' + 'Train_P/'
if not os.path.exists(recorder_log_dir):
    os.makedirs(recorder_log_dir)


class Center_NET(nn.Module):
    def __init__(self, input_dim, hidden_n, output_dim):
        super(Center_NET, self).__init__()
        self.hidden1 = nn.Linear(input_dim, hidden_n)
        self.hidden2 = nn.Linear(hidden_n, output_dim)
    def forward(self, x): # x is the initial weights
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        return nn.Softmax()(x)

def aggregate_tensor(results: List[Tuple[torch.Tensor, int]]) -> List:
    """Compute weighted average using tensor which can transmit gradient information."""
    # Calculate the total number of examples used during training
    num_examples_total = sum(num_examples for (_, num_examples) in results) # 一共有多少个training samples
    num = len(results) # number of clients
    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples / num_examples_total for layer in weights] for weights, num_examples in results
    ]
    convert_list = [] # convert_list is a list and inside this are all tensors.
    for w in weighted_weights: # w is a list
        for l in w:
            convert_list.append(l)
    layer_num = len(convert_list)//num  # layer in each model weights.
    for i in range(layer_num):
        for j in range(num-1):
            convert_list[i] += convert_list[layer_num*(j+1)+i]
    return convert_list[:layer_num]


def distance(para_g:NDArrays, para_i:NDArrays)->float:
    dis = 0
    for key in range(len(para_g)):
        dis += np.linalg.norm(para_g[key] - para_i[key])
    return dis

# n clients as input_dim, n clients as output_dim.
# Softmax is needed because I want to give each



class PriorityStrategy(Strategy):
    def __init__(
            self,
            *,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
            evaluate_fn: Optional[
                Callable[
                    [int, NDArrays, Dict[str, Scalar]],
                    Optional[Tuple[float, Dict[str, Scalar]]],
                ]
            ] = None,
            on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            accept_failures: bool = True,
            initial_parameters: Optional[Parameters] = None,
            global_parameters: Optional[Parameters] = None,
            fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            inplace: bool = False, # 默认是使用自定义的聚合方法的。
            cum_reward_avg : Optional[NDArrays] = None,
            done:bool = False,
            env: MyEnv,
            model: MLP_Q_Net,
            weights: Tensor,
            Center_Net: Center_NET
    ) -> None:
        super().__init__()

        if (
                min_fit_clients > min_available_clients
                or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.global_parameters = initial_parameters # must be updated in each epoch
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.inplace = inplace
        self.cum_reward_avg = np.zeros(record_len)
        self.done = done
        self.env = env
        self.model = model
        self.weights = weights
        self.Center_Net=Center_Net

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        # print("------------------1------------------")
        print("Initializing parameters procedure.!")
        self.model = self.model.to(DEVICE)
        print(next(self.model.parameters()).device)
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        self.global_parameters = initial_parameters
        return initial_parameters

    def evaluate(
            self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        # print("------------------2------------------")
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)

        # server端的评估方法；
        # 需要重写server端的评估方法： -> reward:float
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None

        # 只返回一个reward.
        reward = eval_res
        # loss, metrics = eval_res
        return reward

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        # print("------------------3------------------")
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # wait until all clients are available
        print("STILL WAITING......")
        client_manager.wait_for(N, 10) # 在server端更新所有的参数。

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        print(f"------------This time I am about to sample {sample_size} Clients!-----------------------")
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]
    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # print("------------------4------------------")
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        # 将穿回的global parameters和config一起打包为evaluate_ins
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        # print("------------------5------------------")

        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # 返回的内容是参数的List，但是不太清楚是否会提供fit_res.parameters和fit_res.num_examples.
        if self.inplace:
            # Does in-place weighted average of results
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            # ADD Validation Environment to better aggregate the policy.
            '''evaluate_round = 5  # Can be adjusted according to the aggregation time. 不断更新当前的模型，最多会执行evaluate_round轮的游戏。
            weight_net = self.Center_Net  # parameters. not NDArray

            # LOAD GLOBAL MODEL
            self.weights = torch.tensor(self.weights,requires_grad=True).to(DEVICE) # weight initialization
            weight_after = weight_net(self.weights).cpu().detach().numpy() # weights for client network.
            print(f"--------------------------WEIGHTS 1: {weight_after}-----------------------------")

            parameters = [
                parameters_to_ndarrays(fit_res.parameters)
                for _, fit_res in results
            ]
            weights_results = list(zip(parameters, weight_after))
            aggregated_ndarrays = aggregate(weights_results)
            # initialize the global_net
            params_dict=zip(self.model.state_dict().keys(), aggregated_ndarrays)
            state_dict=OrderedDict({k:torch.tensor(v,requires_grad=True) for k, v in params_dict})
            self.model.load_state_dict(state_dict,strict=True)


            # set up the tgt_net
            tgt_net = ptan.agent.TargetNet(self.model)
            selector = ptan.actions.ArgmaxActionSelector()
            selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=init_epsilon, selector=selector)
            agent = ptan.agent.DQNAgent(self.model, selector, device=DEVICE)
            exp_source = ptan.experience.ExperienceSourceFirstLast(self.env, agent, gamma=gamma)
            replay_buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=replay_size)
            optimizer = optim.Adam(model.parameters(), lr=lr) # Our target net is Center_Net, not "model"!
            optimizer_back = optim.Adam(weight_net.parameters(),lr=lr)

            step = 0
            evaluate_reward = 0
            solved = False
            episode = 0

            while True:

                step += 1
                replay_buffer.populate(1)

                for reward, steps in exp_source.pop_rewards_steps():  # episode执行完后才会返回该episode积累的经验。
                    episode += 1
                    # print("The current episode is: " + str(episode))
                    evaluate_reward += reward
                    if episode % evaluate_episodes_for_conv == 0:
                        evaluate_reward /= evaluate_episodes_for_conv
                        solved = evaluate_reward > solve_criterion
                        evaluate_reward = 0
                if solved:
                    break
                if episode >= evaluate_round:
                    break
                if len(replay_buffer) < 2 * batch_size:
                    continue

                ############### FIRST PERIOD ##############
                batch = replay_buffer.sample(batch_size)
                states_v, actions_v, tgt_q_v = unpack_batch(batch, tgt_net.target_model, gamma)
                optimizer.zero_grad()
                q_v = self.model(states_v)
                q_v = q_v.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
                loss_v = F.mse_loss(q_v, tgt_q_v)
                loss_v.backward() # update parameters in model net
                optimizer.step()
                # 此时model已经被更新了 ---- 更新后的model是 model fusion target

                ############### SECOND PERIOD ##############    update Center_Net()
                # 反向更新了model的参数，然后此时model的参数作为我们更新的目标，进一步计算Center_Net的loss然后更新。
                params_after_update = list(model.parameters()) # our target model parameters
                params_tensor = torch.cat([param.view(-1) for param in params_after_update])# target

                optimizer_back.zero_grad()
                weight = torch.tensor(weight_after, requires_grad=True).to(DEVICE)
                weight_after = weight_net(weight)
                # compute loss
                parameters = [
                    parameters_to_ndarrays(fit_res.parameters)
                    for _, fit_res in results
                ]
                tensor_weights = [
                    [torch.tensor(layer, requires_grad=True).to(DEVICE) for layer in weights] for weights in parameters
                ]
                weights_results = list(zip(tensor_weights, weight_after))
                aggregated_weight = aggregate_tensor(weights_results)
                aggregated_weight_tensor = torch.cat([param.view(-1) for param in aggregated_weight])
                loss = F.mse_loss(aggregated_weight_tensor, params_tensor)
                loss.backward()
                optimizer_back.step()

                ############### SECOND PERIOD ##############       update Q-function.
                weight_after = weight_after.cpu().detach().numpy() # pv <- Center_Net(pv-1)
                parameters = [
                    parameters_to_ndarrays(fit_res.parameters)
                    for _, fit_res in results
                ]
                weights_results = list(zip(parameters, weight_after))
                aggregated_ndarrays = aggregate(weights_results)
                params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v, requires_grad=True) for k, v in params_dict})
                self.model.load_state_dict(state_dict, strict=True)

                selector.epsilon *= epsilon_decay

                if episode % tgt_net_sync == 0:
                    tgt_net.sync()

            # After evaluation, we can get the parameters we want to use through the weight_net.
            # weight_after i pV in timestep t
            print(f"--------------------------WEIGHTS 2: {weight_after}-----------------------------")
            # self.weights = weight_after # do not update global initial parameters.
            weights_results = list(zip(parameters, weight_after))'''

            # 使用client和global模型之间的距离进行聚合，更改weighted_results
            # P_DQN with epsilon-greedy
            '''epsilon= 0.9
            parameters = [
                parameters_to_ndarrays(fit_res.parameters)
                for _, fit_res in results
            ]
            # epsilon * 1.0 /
            weights = [pow(distance(parameters_to_ndarrays(self.global_parameters), net_para), BETA) for net_para in parameters]
            parameters.append(parameters_to_ndarrays(self.global_parameters))
            weights.append(1 - epsilon)
            weights_results = list(zip(parameters, weights))'''

            # R_DQN
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            # DQNAvg
            '''weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), 1)
                for _, fit_res in results
            ]'''

            # 在这里使用了相同的聚合策略
            aggregated_ndarrays = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
        self.global_parameters = parameters_aggregated # update global parameters.
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def write_log(self):
        print("Writing log file!")
        np.save(recorder_log_dir + 'cum_reward_avg.npy', self.cum_reward_avg)
        return

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        # print("------------------6------------------")
        # 在server端如何进行聚合从clients传来的evaluation results.
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # 使用平均法simple average
        SimpleAverage_reward = np.average([evaluate_res.loss for _, evaluate_res in results])

        self.cum_reward_avg[server_round-1] = SimpleAverage_reward
        print(self.cum_reward_avg)
        if server_round == record_len:
            self.done = True
            self.write_log()
        # Aggregate loss
        '''loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.reward)
                for _, evaluate_res in results
            ]
        )'''
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return SimpleAverage_reward, metrics_aggregated

seed = 666
seed_everything(seed)

server_theta = np.random.uniform()
env_train = env_class(para=server_theta)
obs_size = env_train.observation_space.shape[0]
n_actions = env_train.action_space.n  # action_space.n
model = FL_agent.MLP_Q_Net(obs_size, n_actions, **net_extra_hyperparameters).to(DEVICE).eval()
center_net = Center_NET(N, 128, N).to(DEVICE)

print(f"\n________________________{model}_______________________ LOOK AT HERE!")
weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
init_parameters = fl.common.ndarrays_to_parameters(weights)

# Define strategy
strategy = PriorityStrategy(initial_parameters=init_parameters,
                            env=env_train,
                            model=model,
                            weights=torch.tensor([1.0/N for i in range(N)],requires_grad=True),
                            Center_Net=center_net)


# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=outer_loop_iter_upper_bound),
    strategy=strategy,
)