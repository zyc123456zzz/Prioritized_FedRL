import numpy as np
from Sharing import *
import matplotlib.pyplot as plt
import os

# MountainCar environment
'''folder_path_E = '/opt/data/private/P_FedRL/E_DQNAvg-MountainCar-Distributed/Server/'
folder_path_R = '/opt/data/private/P_FedRL/R_DQNAvg-MountainCar-Distributed/Server/'
folder_path_P = '/opt/data/private/P_FedRL/P_DQNAvg-MountainCar-Distributed/Server/'
folder_path = '/opt/data/private/P_FedRL/DQNAvg-MountainCar-Distributed/Server/'
folder_path_P_M = '/opt/data/private/P_FedRL/P_Momentum_DQNAvg-MountainCar-Distributed/Server/'

image_E=[]
image_R=[]
image_P=[]
image=[]
image_P_M=[]

image_E_test=[]
image_R_test=[]
image_P_test=[]
image_test=[]
image_P_M_test=[]

# 遍历文件夹
for root, dirs, files in os.walk(folder_path_E):
    for file in files:
        if file == 'cum_reward_avg.npy':
            file_path = os.path.join(root, file)
            # 在这里可以对每个文件进行处理
            image_E.append(np.load(file_path))
        if file == 'server_test.npy':
            file_path = os.path.join(root, file)
            image_E_test.append(np.load(file_path))
image_E = np.average(image_E, axis=0)
image_E_test=np.average(image_E_test,axis=0)
E12ratio = merge_interval // meta_merge
image_reshaped_E = np.zeros(outer_loop_iter_upper_bound*E12ratio)
image_reshaped_E_test = np.zeros(outer_loop_iter_upper_bound*E12ratio)

for root, dirs, files in os.walk(folder_path_R):
    for file in files:
        if file == 'cum_reward_avg.npy':
            file_path = os.path.join(root, file)
            # 在这里可以对每个文件进行处理
            image_R.append(np.load(file_path))
        if file == 'server_test.npy':
            file_path = os.path.join(root, file)
            image_R_test.append(np.load(file_path))
image_R = np.average(image_R, axis=0)
image_R_test = np.average(image_R_test, axis=0)
E12ratio = merge_interval // meta_merge
image_reshaped_R = np.zeros(outer_loop_iter_upper_bound*E12ratio)
image_reshaped_R_test = np.zeros(outer_loop_iter_upper_bound*E12ratio)

# 遍历文件夹
for root, dirs, files in os.walk(folder_path_P):
    for file in files:
        if file == 'cum_reward_avg.npy':
            file_path = os.path.join(root, file)
            # 在这里可以对每个文件进行处理
            image_P.append(np.load(file_path))
        if file == 'server_test.npy':
            file_path = os.path.join(root, file)
            # 在这里可以对每个文件进行处理
            image_P_test.append(np.load(file_path))
image_P = np.average(image_P, axis=0)
image_P_test=np.average(image_P_test, axis=0)
E12ratio = merge_interval // meta_merge
image_reshaped_P = np.zeros(outer_loop_iter_upper_bound*E12ratio)
image_reshaped_P_test = np.zeros(outer_loop_iter_upper_bound*E12ratio)

# 遍历文件夹
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file == 'cum_reward_avg.npy':
            file_path = os.path.join(root, file)
            # 在这里可以对每个文件进行处理
            image.append(np.load(file_path))
        if file == 'server_test.npy':
            file_path = os.path.join(root, file)
            # 在这里可以对每个文件进行处理
            image_test.append(np.load(file_path))
image = np.average(image, axis=0)
image_test=np.average(image_test, axis=0)
E12ratio = merge_interval // meta_merge
image_reshaped = np.zeros(outer_loop_iter_upper_bound*E12ratio)
image_reshaped_test = np.zeros(outer_loop_iter_upper_bound*E12ratio)


for root, dirs, files in os.walk(folder_path_P_M):
    for file in files:
        if file == 'cum_reward_avg.npy':
            file_path = os.path.join(root, file)
            # 在这里可以对每个文件进行处理
            image_P_M.append(np.load(file_path))
        if file == 'server_test.npy':
            file_path = os.path.join(root, file)
            # 在这里可以对每个文件进行处理
            image_P_M_test.append(np.load(file_path))
image_P_M = np.average(image_P_M, axis=0)
image_P_M_test = np.average(image_P_M_test,axis=0)
E12ratio = merge_interval // meta_merge
image_reshaped_P_M = np.zeros(outer_loop_iter_upper_bound*E12ratio)
image_reshaped_P_M_test = np.zeros(outer_loop_iter_upper_bound*E12ratio)

# reshape the output so that the plot is nicer.
for i in range(outer_loop_iter_upper_bound):
    image_reshaped_E[i * E12ratio:(i + 1) * E12ratio] = image_E[i]
    image_reshaped_R[i * E12ratio:(i + 1) * E12ratio] = image_R[i]
    image_reshaped_P[i * E12ratio:(i + 1) * E12ratio] = image_P[i]
    image_reshaped[i * E12ratio:(i + 1) * E12ratio] = image[i]
    image_reshaped_P_M[i * E12ratio:(i + 1) * E12ratio] = image_P_M[i]
    image_reshaped_E_test[i * E12ratio:(i + 1) * E12ratio] = image_E_test[i]
    image_reshaped_R_test[i * E12ratio:(i + 1) * E12ratio] = image_R_test[i]
    image_reshaped_P_test[i * E12ratio:(i + 1) * E12ratio] = image_P_test[i]
    image_reshaped_test[i * E12ratio:(i + 1) * E12ratio] = image_test[i]
    image_reshaped_P_M_test[i * E12ratio:(i + 1) * E12ratio] = image_P_M_test[i]

y = image_reshaped_E
y0 = image_reshaped_R
y1 = image_reshaped_P
y2 = image_reshaped
y11 = image_reshaped_P_M

y_test = image_reshaped_E_test
y0_test = image_reshaped_R_test
y1_test = image_reshaped_P_test
y2_test = image_reshaped_test
y11_test = image_reshaped_P_M_test

x = np.arange(0, train_episodes_upper_bound, train_episodes_upper_bound//(outer_loop_iter_upper_bound*E12ratio))
plt.figure()

plt.plot(x, y, color='red', label='E_DQNAvg')
plt.plot(x, y0, color='green', label='R_DQNAvg')
plt.plot(x, y1, color='blue', label='P_DQNAvg')
# plt.plot(x, y11, color='purple', label='P_Momentum_DQNAvg')
plt.plot(x, y2, color='orange', label='DQNAvg')

# 图例位置， 坐标名称
plt.title('MountainCar')
plt.legend(loc='upper left')
plt.xlabel('Episodes')
plt.ylabel('Obj Value')

# plt.subplot(2,1,2)
plt.plot(x, y_test, color='red', label='E_DQNAvg')
plt.plot(x, y0_test, color='green', label='R_DQNAvg')
plt.plot(x, y1_test, color='blue', label='P_DQNAvg')
# plt.plot(x, y11_test, color='purple', label='P_Momentum_DQNAvg')
plt.plot(x, y2_test, color='orange', label='DQNAvg')

# 图例位置， 坐标名称
plt.title('MountainCar')
plt.legend(loc='upper left')
plt.xlabel('Episodes')
plt.ylabel('Generalization Obj Value')


plt.savefig('/opt/data/private/P_FedRL/plot-DQNAvg-MountainCar-Distributed/MountainCar_Compare.png')
plt.show()'''

folder_path_2 = '/opt/data/private/P_FedRL/P_DQNAvg-Acrobot-Distributed/Server/client_number_is2/'
folder_path_5 = '/opt/data/private/P_FedRL/P_DQNAvg-Acrobot-Distributed/Server/client_number_is5/'
folder_path_10 = '/opt/data/private/P_FedRL/P_DQNAvg-Acrobot-Distributed/Server/client_number_is10/'
folder_path_50 = '/opt/data/private/P_FedRL/P_DQNAvg-Acrobot-Distributed/Server/client_number_is50/'

image_2=[]
image_5=[]
image_10=[]
image_50=[]

image_2_test=[]
image_5_test=[]
image_10_test=[]
image_50_test=[]

# 遍历文件夹
for root, dirs, files in os.walk(folder_path_2):
    for file in files:
        print(file)
        if file == 'cum_reward_avg.npy':
            file_path = os.path.join(root, file)
            # 在这里可以对每个文件进行处理
            image_2.append(np.load(file_path))
        if file == 'server_test.npy':
            file_path = os.path.join(root, file)
            image_2_test.append(np.load(file_path))
image_2 = np.average(image_2, axis=0)
image_2_test=np.average(image_2_test,axis=0)
E12ratio = merge_interval // meta_merge
image_reshaped_2 = np.zeros(outer_loop_iter_upper_bound*E12ratio)
image_reshaped_2_test = np.zeros(outer_loop_iter_upper_bound*E12ratio)

for root, dirs, files in os.walk(folder_path_5):
    for file in files:
        if file == 'cum_reward_avg.npy':
            file_path = os.path.join(root, file)
            # 在这里可以对每个文件进行处理
            image_5.append(np.load(file_path))
        if file == 'server_test.npy':
            file_path = os.path.join(root, file)
            image_5_test.append(np.load(file_path))
image_5 = np.average(image_5, axis=0)
image_5_test = np.average(image_5_test, axis=0)
E12ratio = merge_interval // meta_merge
image_reshaped_5 = np.zeros(outer_loop_iter_upper_bound*E12ratio)
image_reshaped_5_test = np.zeros(outer_loop_iter_upper_bound*E12ratio)

# 遍历文件夹
for root, dirs, files in os.walk(folder_path_10):
    for file in files:
        if file == 'cum_reward_avg.npy':
            file_path = os.path.join(root, file)
            # 在这里可以对每个文件进行处理
            image_10.append(np.load(file_path))
        if file == 'server_test.npy':
            file_path = os.path.join(root, file)
            # 在这里可以对每个文件进行处理
            image_10_test.append(np.load(file_path))
image_10 = np.average(image_10, axis=0)
image_10_test=np.average(image_10_test, axis=0)
E12ratio = merge_interval // meta_merge
image_reshaped_10 = np.zeros(outer_loop_iter_upper_bound*E12ratio)
image_reshaped_10_test = np.zeros(outer_loop_iter_upper_bound*E12ratio)

for root, dirs, files in os.walk(folder_path_50):
    for file in files:
        if file == 'cum_reward_avg.npy':
            file_path = os.path.join(root, file)
            # 在这里可以对每个文件进行处理
            image_50.append(np.load(file_path))
        if file == 'server_test.npy':
            file_path = os.path.join(root, file)
            # 在这里可以对每个文件进行处理
            image_50_test.append(np.load(file_path))
image_50 = np.average(image_50, axis=0)
image_50_test = np.average(image_50_test,axis=0)
E12ratio = merge_interval // meta_merge
image_reshaped_50 = np.zeros(outer_loop_iter_upper_bound*E12ratio)
image_reshaped_50_test = np.zeros(outer_loop_iter_upper_bound*E12ratio)

print(image_reshaped_2)

# reshape the output so that the plot is nicer.
for i in range(outer_loop_iter_upper_bound):
    image_reshaped_2[i * E12ratio:(i + 1) * E12ratio] = image_2[i]
    image_reshaped_5[i * E12ratio:(i + 1) * E12ratio] = image_5[i]
    image_reshaped_10[i * E12ratio:(i + 1) * E12ratio] = image_10[i]
    image_reshaped_50[i * E12ratio:(i + 1) * E12ratio] = image_50[i]

    image_reshaped_2_test[i * E12ratio:(i + 1) * E12ratio] = image_2_test[i]
    image_reshaped_5_test[i * E12ratio:(i + 1) * E12ratio] = image_5_test[i]
    image_reshaped_10_test[i * E12ratio:(i + 1) * E12ratio] = image_10_test[i]
    image_reshaped_50_test[i * E12ratio:(i + 1) * E12ratio] = image_50_test[i]

y = image_reshaped_2
y0 = image_reshaped_5
y1 = image_reshaped_10
y11 = image_reshaped_50

y_test = image_reshaped_2_test
y0_test = image_reshaped_5_test
y1_test = image_reshaped_10_test
y11_test = image_reshaped_50_test

x = np.arange(0, train_episodes_upper_bound, train_episodes_upper_bound//(outer_loop_iter_upper_bound*E12ratio))
plt.figure()

'''plt.plot(x, y, color='red', label='N=2')
plt.plot(x, y0, color='green', label='N=5')
plt.plot(x, y1, color='blue', label='N=10')
plt.plot(x, y11, color='purple', label='N=50')

# 图例位置， 坐标名称
plt.title('Acrobot')
plt.legend(loc='upper left')
plt.xlabel('Episodes')
plt.ylabel('Obj Value')'''

plt.plot(x, y_test, color='red', label='N=2')
plt.plot(x, y0_test, color='green', label='N=5')
plt.plot(x, y1_test, color='blue', label='N=10')
plt.plot(x, y11_test, color='purple', label='N=50')

# 图例位置， 坐标名称
plt.title('Acrobot')
plt.legend(loc='upper left')
plt.xlabel('Episodes')
plt.ylabel('Generalization Obj Value')

plt.savefig('/opt/data/private/P_FedRL/plot-DQNAvg-Acrobot-Distributed/Acrobot_Compare_N_P.png')
plt.show()



# Acrobot example
'''
folder_path_E = '/opt/data/private/P_FedRL/E_DQNAvg-Acrobot-Distributed/Server/'
folder_path_R = '/opt/data/private/P_FedRL/R_DQNAvg-Acrobot-Distributed/Server/'
folder_path_P = '/opt/data/private/P_FedRL/P_DQNAvg-Acrobot-Distributed/Server/'
folder_path = '/opt/data/private/P_FedRL/DQNAvg-Acrobot-Distributed/Server/'
folder_path_P_M = '/opt/data/private/P_FedRL/P_Momentum_DQNAvg-Acrobot-Distributed/Server/'

image_E=[]
image_R=[]
image_P=[]
image=[]
image_P_M=[]

image_E_test=[]
image_R_test=[]
image_P_test=[]
image_test=[]
image_P_M_test=[]

# 遍历文件夹
for root, dirs, files in os.walk(folder_path_E):
    for file in files:
        if file == 'cum_reward_avg.npy':
            file_path = os.path.join(root, file)
            # 在这里可以对每个文件进行处理
            image_E.append(np.load(file_path))
        if file == 'server_test.npy':
            file_path = os.path.join(root, file)
            image_E_test.append(np.load(file_path))
image_E = np.average(image_E, axis=0)
image_E_test=np.average(image_E_test,axis=0)
E12ratio = merge_interval // meta_merge
image_reshaped_E = np.zeros(outer_loop_iter_upper_bound*E12ratio)
image_reshaped_E_test = np.zeros(outer_loop_iter_upper_bound*E12ratio)

for root, dirs, files in os.walk(folder_path_R):
    for file in files:
        if file == 'cum_reward_avg.npy':
            file_path = os.path.join(root, file)
            # 在这里可以对每个文件进行处理
            image_R.append(np.load(file_path))
        if file == 'server_test.npy':
            file_path = os.path.join(root, file)
            image_R_test.append(np.load(file_path))
image_R = np.average(image_R, axis=0)
image_R_test = np.average(image_R_test, axis=0)
E12ratio = merge_interval // meta_merge
image_reshaped_R = np.zeros(outer_loop_iter_upper_bound*E12ratio)
image_reshaped_R_test = np.zeros(outer_loop_iter_upper_bound*E12ratio)

# 遍历文件夹
for root, dirs, files in os.walk(folder_path_P):
    for file in files:
        if file == 'cum_reward_avg.npy':
            file_path = os.path.join(root, file)
            # 在这里可以对每个文件进行处理
            image_P.append(np.load(file_path))
        if file == 'server_test.npy':
            file_path = os.path.join(root, file)
            # 在这里可以对每个文件进行处理
            image_P_test.append(np.load(file_path))
image_P = np.average(image_P, axis=0)
image_P_test=np.average(image_P_test, axis=0)
E12ratio = merge_interval // meta_merge
image_reshaped_P = np.zeros(outer_loop_iter_upper_bound*E12ratio)
image_reshaped_P_test = np.zeros(outer_loop_iter_upper_bound*E12ratio)

# 遍历文件夹
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file == 'cum_reward_avg.npy':
            file_path = os.path.join(root, file)
            # 在这里可以对每个文件进行处理
            image.append(np.load(file_path))
        if file == 'server_test.npy':
            file_path = os.path.join(root, file)
            # 在这里可以对每个文件进行处理
            image_test.append(np.load(file_path))
image = np.average(image, axis=0)
image_test=np.average(image_test, axis=0)
E12ratio = merge_interval // meta_merge
image_reshaped = np.zeros(outer_loop_iter_upper_bound*E12ratio)
image_reshaped_test = np.zeros(outer_loop_iter_upper_bound*E12ratio)


for root, dirs, files in os.walk(folder_path_P_M):
    for file in files:
        if file == 'cum_reward_avg.npy':
            file_path = os.path.join(root, file)
            # 在这里可以对每个文件进行处理
            image_P_M.append(np.load(file_path))
        if file == 'server_test.npy':
            file_path = os.path.join(root, file)
            # 在这里可以对每个文件进行处理
            image_P_M_test.append(np.load(file_path))
image_P_M = np.average(image_P_M, axis=0)
image_P_M_test = np.average(image_P_M_test,axis=0)
E12ratio = merge_interval // meta_merge
image_reshaped_P_M = np.zeros(outer_loop_iter_upper_bound*E12ratio)
image_reshaped_P_M_test = np.zeros(outer_loop_iter_upper_bound*E12ratio)

# reshape the output so that the plot is nicer.
for i in range(outer_loop_iter_upper_bound):
    image_reshaped_E[i * E12ratio:(i + 1) * E12ratio] = image_E[i]
    image_reshaped_R[i * E12ratio:(i + 1) * E12ratio] = image_R[i]
    image_reshaped_P[i * E12ratio:(i + 1) * E12ratio] = image_P[i]
    image_reshaped[i * E12ratio:(i + 1) * E12ratio] = image[i]
    image_reshaped_P_M[i * E12ratio:(i + 1) * E12ratio] = image_P_M[i]

    image_reshaped_E_test[i * E12ratio:(i + 1) * E12ratio] = image_E_test[i]
    image_reshaped_R_test[i * E12ratio:(i + 1) * E12ratio] = image_R_test[i]
    image_reshaped_P_test[i * E12ratio:(i + 1) * E12ratio] = image_P_test[i]
    image_reshaped_test[i * E12ratio:(i + 1) * E12ratio] = image_test[i]
    image_reshaped_P_M_test[i * E12ratio:(i + 1) * E12ratio] = image_P_M_test[i]

y = image_reshaped_E
y0 = image_reshaped_R
y1 = image_reshaped_P
y2 = image_reshaped
y11 = image_reshaped_P_M

y_test = image_reshaped_E_test
y0_test = image_reshaped_R_test
y1_test = image_reshaped_P_test
y2_test = image_reshaped_test
y11_test = image_reshaped_P_M_test

x = np.arange(0, train_episodes_upper_bound, train_episodes_upper_bound//(outer_loop_iter_upper_bound*E12ratio))
plt.figure()

plt.plot(x, y, color='red', label='E_DQNAvg')
plt.plot(x, y0, color='green', label='R_DQNAvg')
plt.plot(x, y1, color='blue', label='P_DQNAvg')
# plt.plot(x, y11, color='purple', label='P_Momentum_DQNAvg')
plt.plot(x, y2, color='orange', label='DQNAvg')

# 图例位置， 坐标名称
plt.title('Acrobot')
plt.legend(loc='upper left')
plt.xlabel('Episodes')
plt.ylabel('Obj Value')

plt.plot(x, y_test, color='red', label='E_DQNAvg')
plt.plot(x, y0_test, color='green', label='R_DQNAvg')
plt.plot(x, y1_test, color='blue', label='P_DQNAvg')
# plt.plot(x, y11_test, color='purple', label='P_Momentum_DQNAvg')
plt.plot(x, y2_test, color='orange', label='DQNAvg')

# 图例位置， 坐标名称
plt.title('Acrobot')
plt.legend(loc='upper left')
plt.xlabel('Episodes')
plt.ylabel('Generalization Obj Value')

plt.savefig('/opt/data/private/P_FedRL/plot-DQNAvg-Acrobot-Distributed/Acrobot_Compare.png')
plt.show()'''


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

image_E_test=[]
image_R_test=[]
image_P_test=[]
image_test=[]
image_P_M_test=[]

# 遍历文件夹
for root, dirs, files in os.walk(folder_path_E):
    for file in files:
        if file == 'cum_reward_avg.npy':
            file_path = os.path.join(root, file)
            # 在这里可以对每个文件进行处理
            image_E.append(np.load(file_path))
        if file == 'server_test.npy':
            file_path = os.path.join(root, file)
            image_E_test.append(np.load(file_path))
image_E = np.average(image_E, axis=0)
image_E_test=np.average(image_E_test,axis=0)
E12ratio = merge_interval // meta_merge
image_reshaped_E = np.zeros(outer_loop_iter_upper_bound*E12ratio)
image_reshaped_E_test = np.zeros(outer_loop_iter_upper_bound*E12ratio)

for root, dirs, files in os.walk(folder_path_R):
    for file in files:
        if file == 'cum_reward_avg.npy':
            file_path = os.path.join(root, file)
            # 在这里可以对每个文件进行处理
            image_R.append(np.load(file_path))
        if file == 'server_test.npy':
            file_path = os.path.join(root, file)
            image_R_test.append(np.load(file_path))
image_R = np.average(image_R, axis=0)
image_R_test = np.average(image_R_test, axis=0)
E12ratio = merge_interval // meta_merge
image_reshaped_R = np.zeros(outer_loop_iter_upper_bound*E12ratio)
image_reshaped_R_test = np.zeros(outer_loop_iter_upper_bound*E12ratio)

# 遍历文件夹
for root, dirs, files in os.walk(folder_path_P):
    for file in files:
        if file == 'cum_reward_avg.npy':
            file_path = os.path.join(root, file)
            # 在这里可以对每个文件进行处理
            image_P.append(np.load(file_path))
        if file == 'server_test.npy':
            file_path = os.path.join(root, file)
            # 在这里可以对每个文件进行处理
            image_P_test.append(np.load(file_path))
image_P = np.average(image_P, axis=0)
image_P_test=np.average(image_P_test, axis=0)
E12ratio = merge_interval // meta_merge
image_reshaped_P = np.zeros(outer_loop_iter_upper_bound*E12ratio)
image_reshaped_P_test = np.zeros(outer_loop_iter_upper_bound*E12ratio)

# 遍历文件夹
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file == 'cum_reward_avg.npy':
            file_path = os.path.join(root, file)
            # 在这里可以对每个文件进行处理
            image.append(np.load(file_path))
        if file == 'server_test.npy':
            file_path = os.path.join(root, file)
            # 在这里可以对每个文件进行处理
            image_test.append(np.load(file_path))
image = np.average(image, axis=0)
image_test=np.average(image_test, axis=0)
E12ratio = merge_interval // meta_merge
image_reshaped = np.zeros(outer_loop_iter_upper_bound*E12ratio)
image_reshaped_test = np.zeros(outer_loop_iter_upper_bound*E12ratio)


for root, dirs, files in os.walk(folder_path_P_M):
    for file in files:
        if file == 'cum_reward_avg.npy':
            file_path = os.path.join(root, file)
            # 在这里可以对每个文件进行处理
            image_P_M.append(np.load(file_path))
        if file == 'server_test.npy':
            file_path = os.path.join(root, file)
            # 在这里可以对每个文件进行处理
            image_P_M_test.append(np.load(file_path))
image_P_M = np.average(image_P_M, axis=0)
image_P_M_test = np.average(image_P_M_test,axis=0)
E12ratio = merge_interval // meta_merge
image_reshaped_P_M = np.zeros(outer_loop_iter_upper_bound*E12ratio)
image_reshaped_P_M_test = np.zeros(outer_loop_iter_upper_bound*E12ratio)

# reshape the output so that the plot is nicer.
for i in range(outer_loop_iter_upper_bound):
    image_reshaped_E[i * E12ratio:(i + 1) * E12ratio] = image_E[i]
    image_reshaped_R[i * E12ratio:(i + 1) * E12ratio] = image_R[i]
    image_reshaped_P[i * E12ratio:(i + 1) * E12ratio] = image_P[i]
    image_reshaped[i * E12ratio:(i + 1) * E12ratio] = image[i]
    image_reshaped_P_M[i * E12ratio:(i + 1) * E12ratio] = image_P_M[i]
    image_reshaped_E_test[i * E12ratio:(i + 1) * E12ratio] = image_E_test[i]
    image_reshaped_R_test[i * E12ratio:(i + 1) * E12ratio] = image_R_test[i]
    image_reshaped_P_test[i * E12ratio:(i + 1) * E12ratio] = image_P_test[i]
    image_reshaped_test[i * E12ratio:(i + 1) * E12ratio] = image_test[i]
    image_reshaped_P_M_test[i * E12ratio:(i + 1) * E12ratio] = image_P_M_test[i]

y = image_reshaped_E
y0 = image_reshaped_R
y1 = image_reshaped_P
y2 = image_reshaped
y11 = image_reshaped_P_M

y_test = image_reshaped_E_test
y0_test = image_reshaped_R_test
y1_test = image_reshaped_P_test
y2_test = image_reshaped_test
y11_test = image_reshaped_P_M_test

x = np.arange(0, train_episodes_upper_bound, train_episodes_upper_bound//(outer_loop_iter_upper_bound*E12ratio))
plt.figure()

plt.plot(x, y, color='red', label='E_DQNAvg')
plt.plot(x, y0, color='green', label='R_DQNAvg')
plt.plot(x, y1, color='blue', label='P_DQNAvg')
# plt.plot(x, y11, color='purple', label='P_Momentum_DQNAvg')
plt.plot(x, y2, color='orange', label='DQNAvg')

# 图例位置， 坐标名称
plt.title('Cartpole')
plt.legend(loc='lower right')
plt.xlabel('Episodes')
plt.ylabel('Obj Value')

plt.plot(x, y_test, color='red', label='E_DQNAvg')
plt.plot(x, y0_test, color='green', label='R_DQNAvg')
plt.plot(x, y1_test, color='blue', label='P_DQNAvg')
# plt.plot(x, y11_test, color='purple', label='P_Momentum_DQNAvg')
plt.plot(x, y2_test, color='orange', label='DQNAvg')

# 图例位置， 坐标名称
plt.title('Cartpole')
plt.legend(loc='lower right')
plt.xlabel('Episodes')
plt.ylabel('Generalization Obj Value')

plt.savefig('/opt/data/private/P_FedRL/plot-DQNAvg-CartPole-Distributed/CartPole_Compare.png')
plt.show()'''