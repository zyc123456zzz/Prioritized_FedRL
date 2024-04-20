import numpy as np
from Sharing import *
import matplotlib.pyplot as plt
import os

folder_path_E = '/opt/data/private/P_FedRL/E_DQNAvg-MountainCar-Distributed/Server/'
folder_path_R = '/opt/data/private/P_FedRL/R_DQNAvg-MountainCar-Distributed/Server/'
folder_path_P = '/opt/data/private/P_FedRL/P_DQNAvg-MountainCar-Distributed/Server/'
folder_path = '/opt/data/private/P_FedRL/DQNAvg-MountainCar-Distributed/Server/'
folder_path_P_M = '/opt/data/private/P_FedRL/P_Momentum_DQNAvg-MountainCar-Distributed/Server/'

image_E=[]
image_R=[]
image_P=[]
image=[]
image_P_M=[]

# 遍历文件夹
for root, dirs, files in os.walk(folder_path_E):
    for file in files:
        file_path = os.path.join(root, file)
        # 在这里可以对每个文件进行处理
        image_E.append(np.load(file_path))
image_E = np.average(image_E, axis=0)
E12ratio = merge_interval // meta_merge
image_reshaped_E = np.zeros(outer_loop_iter_upper_bound*E12ratio)

for root, dirs, files in os.walk(folder_path_R):
    for file in files:
        file_path = os.path.join(root, file)
        # 在这里可以对每个文件进行处理
        image_R.append(np.load(file_path))
image_R = np.average(image_R, axis=0)
E12ratio = merge_interval // meta_merge
image_reshaped_R = np.zeros(outer_loop_iter_upper_bound*E12ratio)

# 遍历文件夹
for root, dirs, files in os.walk(folder_path_P):
    for file in files:
        file_path = os.path.join(root, file)
        # 在这里可以对每个文件进行处理
        image_P.append(np.load(file_path))
image_P = np.average(image_P, axis=0)
E12ratio = merge_interval // meta_merge
image_reshaped_P = np.zeros(outer_loop_iter_upper_bound*E12ratio)

# 遍历文件夹
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        # 在这里可以对每个文件进行处理
        image.append(np.load(file_path))
image = np.average(image, axis=0)
E12ratio = merge_interval // meta_merge
image_reshaped = np.zeros(outer_loop_iter_upper_bound*E12ratio)


for root, dirs, files in os.walk(folder_path_P_M):
    for file in files:
        file_path = os.path.join(root, file)
        # 在这里可以对每个文件进行处理
        image_P_M.append(np.load(file_path))
image_P_M = np.average(image_P_M, axis=0)
E12ratio = merge_interval // meta_merge
image_reshaped_P_M = np.zeros(outer_loop_iter_upper_bound*E12ratio)

# reshape the output so that the plot is nicer.
for i in range(outer_loop_iter_upper_bound):
    image_reshaped_E[i * E12ratio:(i + 1) * E12ratio] = image_E[i]
    image_reshaped_R[i * E12ratio:(i + 1) * E12ratio] = image_R[i]
    image_reshaped_P[i * E12ratio:(i + 1) * E12ratio] = image_P[i]
    image_reshaped[i * E12ratio:(i + 1) * E12ratio] = image[i]
    image_reshaped_P_M[i * E12ratio:(i + 1) * E12ratio] = image_P_M[i]

y = image_reshaped_E
y0 = image_reshaped_R
y1 = image_reshaped_P
y2 = image_reshaped
y11 = image_reshaped_P_M


x = np.arange(0, train_episodes_upper_bound, train_episodes_upper_bound//(outer_loop_iter_upper_bound*E12ratio))
plt.figure()
plt.plot(x, y, color='red', label='E_DQNAvg')
plt.plot(x, y0, color='green', label='R_DQNAvg')
plt.plot(x, y1, color='blue', label='P_DQNAvg')
# plt.plot(x, y11, color='purple', label='P_Momentum_DQNAvg')
plt.plot(x, y2, color='orange', label='DQNAvg')

# 图例位置， 坐标名称
plt.title('MountainCar')
plt.legend(loc='upper right')
plt.xlabel('Episodes')
plt.ylabel('Obj Value')


plt.savefig('/opt/data/private/P_FedRL/plot-DQNAvg-MountainCar-Distributed/MountainCar_Compare.png')
plt.show()


# Acrobot example
'''folder_path_E = '/opt/data/private/P_FedRL/E_DQNAvg-Acrobot-Distributed/Server/'
folder_path_R = '/opt/data/private/P_FedRL/R_DQNAvg-Acrobot-Distributed/Server/'
folder_path_P = '/opt/data/private/P_FedRL/P_DQNAvg-Acrobot-Distributed/Server/'
folder_path = '/opt/data/private/P_FedRL/DQNAvg-Acrobot-Distributed/Server/'
folder_path_P_M = '/opt/data/private/P_FedRL/P_Momentum_DQNAvg-Acrobot-Distributed/Server/'

image_E=[]
image_R=[]
image_P=[]
image=[]
image_P_M=[]

# 遍历文件夹
for root, dirs, files in os.walk(folder_path_E):
    for file in files:
        file_path = os.path.join(root, file)
        # 在这里可以对每个文件进行处理
        image_E.append(np.load(file_path))
image_E = np.average(image_E, axis=0)
E12ratio = merge_interval // meta_merge
image_reshaped_E = np.zeros(outer_loop_iter_upper_bound*E12ratio)

for root, dirs, files in os.walk(folder_path_R):
    for file in files:
        file_path = os.path.join(root, file)
        # 在这里可以对每个文件进行处理
        image_R.append(np.load(file_path))
image_R = np.average(image_R, axis=0)
E12ratio = merge_interval // meta_merge
image_reshaped_R = np.zeros(outer_loop_iter_upper_bound*E12ratio)

# 遍历文件夹
for root, dirs, files in os.walk(folder_path_P):
    for file in files:
        file_path = os.path.join(root, file)
        # 在这里可以对每个文件进行处理
        image_P.append(np.load(file_path))
image_P = np.average(image_P, axis=0)
E12ratio = merge_interval // meta_merge
image_reshaped_P = np.zeros(outer_loop_iter_upper_bound*E12ratio)

# 遍历文件夹
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        # 在这里可以对每个文件进行处理
        image.append(np.load(file_path))
image = np.average(image, axis=0)
E12ratio = merge_interval // meta_merge
image_reshaped = np.zeros(outer_loop_iter_upper_bound*E12ratio)

for root, dirs, files in os.walk(folder_path_P_M):
    for file in files:
        file_path = os.path.join(root, file)
        # 在这里可以对每个文件进行处理
        image_P_M.append(np.load(file_path))
image_P_M = np.average(image_P_M, axis=0)
E12ratio = merge_interval // meta_merge
image_reshaped_P_M = np.zeros(outer_loop_iter_upper_bound*E12ratio)

# reshape the output so that the plot is nicer.
for i in range(outer_loop_iter_upper_bound):
    image_reshaped_E[i * E12ratio:(i + 1) * E12ratio] = image_E[i]
    image_reshaped_R[i * E12ratio:(i + 1) * E12ratio] = image_R[i]
    image_reshaped_P[i * E12ratio:(i + 1) * E12ratio] = image_P[i]
    image_reshaped[i * E12ratio:(i + 1) * E12ratio] = image[i]
    image_reshaped_P_M[i * E12ratio:(i + 1) * E12ratio] = image_P_M[i]

y = image_reshaped_E
y0 = image_reshaped_R
y1 = image_reshaped_P
y2 = image_reshaped
y11 = image_reshaped_P_M


x = np.arange(0, train_episodes_upper_bound, train_episodes_upper_bound//(outer_loop_iter_upper_bound*E12ratio))
plt.figure()
# plt.plot(x, y, color='red', label='E_DQNAvg')
plt.plot(x, y0, color='green', label='R_DQNAvg')
plt.plot(x, y1, color='blue', label='P_DQNAvg')
plt.plot(x, y11, color='purple', label='P_Momentum_DQNAvg')
plt.plot(x, y2, color='orange', label='DQNAvg')

# 图例位置， 坐标名称
plt.title('Acrobot')
plt.legend(loc='upper left')
plt.xlabel('Episodes')
plt.ylabel('Obj Value')


plt.savefig('/opt/data/private/P_FedRL/plot-DQNAvg-Acrobot-Distributed/Acrobot_Compare.png')
plt.show()
'''

# CartPole example
'''folder_path_E = '/opt/data/private/P_FedRL/E_DQNAvg-CartPole-Distributed/Server/'
folder_path_R = '/opt/data/private/P_FedRL/R_DQNAvg-CartPole-Distributed/Server/'
folder_path_P = '/opt/data/private/P_FedRL/P_DQNAvg-CartPole-Distributed/Server/'
folder_path = '/opt/data/private/P_FedRL/DQNAvg-CartPole-Distributed/Server/'
folder_path_P_M = '/opt/data/private/P_FedRL/P_Momentum_DQNAvg-CartPole-Distributed/Server/'

image_E=[]
image_R=[]
image_P=[]
image=[]
image_P_M=[]

# 遍历文件夹
for root, dirs, files in os.walk(folder_path_E):
    for file in files:
        file_path = os.path.join(root, file)
        # 在这里可以对每个文件进行处理
        image_E.append(np.load(file_path))
image_E = np.average(image_E, axis=0)
E12ratio = merge_interval // meta_merge
image_reshaped_E = np.zeros(outer_loop_iter_upper_bound*E12ratio)

for root, dirs, files in os.walk(folder_path_R):
    for file in files:
        file_path = os.path.join(root, file)
        # 在这里可以对每个文件进行处理
        image_R.append(np.load(file_path))
image_R = np.average(image_R, axis=0)
E12ratio = merge_interval // meta_merge
image_reshaped_R = np.zeros(outer_loop_iter_upper_bound*E12ratio)

# 遍历文件夹
for root, dirs, files in os.walk(folder_path_P):
    for file in files:
        file_path = os.path.join(root, file)
        # 在这里可以对每个文件进行处理
        image_P.append(np.load(file_path))
image_P = np.average(image_P, axis=0)
E12ratio = merge_interval // meta_merge
image_reshaped_P = np.zeros(outer_loop_iter_upper_bound*E12ratio)

# 遍历文件夹
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        # 在这里可以对每个文件进行处理
        image.append(np.load(file_path))
image = np.average(image, axis=0)
E12ratio = merge_interval // meta_merge
image_reshaped = np.zeros(outer_loop_iter_upper_bound*E12ratio)

for root, dirs, files in os.walk(folder_path_P_M):
    for file in files:
        file_path = os.path.join(root, file)
        # 在这里可以对每个文件进行处理
        image_P_M.append(np.load(file_path))
image_P_M = np.average(image_P_M, axis=0)
E12ratio = merge_interval // meta_merge
image_reshaped_P_M = np.zeros(outer_loop_iter_upper_bound*E12ratio)

# reshape the output so that the plot is nicer.
for i in range(outer_loop_iter_upper_bound):
    image_reshaped_E[i * E12ratio:(i + 1) * E12ratio] = image_E[i]
    image_reshaped_R[i * E12ratio:(i + 1) * E12ratio] = image_R[i]
    image_reshaped_P[i * E12ratio:(i + 1) * E12ratio] = image_P[i]
    image_reshaped[i * E12ratio:(i + 1) * E12ratio] = image[i]
    image_reshaped_P_M[i * E12ratio:(i + 1) * E12ratio] = image_P_M[i]

y = image_reshaped_E
y0 = image_reshaped_R
y1 = image_reshaped_P
y2 = image_reshaped
y11 = image_reshaped_P_M


x = np.arange(0, train_episodes_upper_bound, train_episodes_upper_bound//(outer_loop_iter_upper_bound*E12ratio))
plt.figure()
# plt.plot(x, y, color='red', label='E_DQNAvg')
plt.plot(x, y0, color='green', label='R_DQNAvg')
plt.plot(x, y1, color='blue', label='P_DQNAvg')
plt.plot(x, y2, color='orange', label='DQNAvg')
plt.plot(x, y11, color='purple', label='P_Momentum_DQNAvg')

# 图例位置， 坐标名称
plt.title('Cartpole')
plt.legend(loc='lower right')
plt.xlabel('Episodes')
plt.ylabel('Obj Value')

plt.savefig('/opt/data/private/P_FedRL/plot-DQNAvg-CartPole-Distributed/CartPole_Compare.png')
plt.show()'''