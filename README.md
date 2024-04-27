## FRL system explanation

### Structure

The whole Federated Reinforcement Learning system consists of two parts:

**Server.py** **Client.py**

1. If we want to train a FRL system, we need bound the exact ip and some other network information for Clients so that it knows how to connect with the server.
2. If we just want to train FRL on my own laptop, remain the settings in the .py files.

This FRL system implemented with Flower is highly modular, you can do modifications to several modules so that you can achieve your own purpose.

#### Module

**FL_agent.py** consists of RL agent training and evaluation methods, and you can modify this file so that you can train and evaluate your own Federated RL agent.

**Server.py** consists of **Strategy** which controls *client initialization*, *client configuration*, *fit aggregation*, *centralized evaluation*, *evaluation aggregation* methods where you can do modifications like model fusion and client selection etc as you want.
