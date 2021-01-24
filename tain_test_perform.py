import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

step = 10

data = pd.read_csv('train_perform.csv', header=None)
perform = ['Size', 'FCT', 'Tput', 'EF_FCT', 'EF_num', 'EF_avgFCT', 'DMR']
data.columns = perform
# print data

DMR = data['DMR']
FCT = data['FCT']
EF_FCT = data['EF_FCT']
EF_avgFCT = data['EF_avgFCT']
DMR_sum = data['DMR'].sum()

episode = data.shape[0] / step
print('episode:', episode)
episode_DMR = []
episode_FCT = []
episode_EF_FCT = []
episode_EF_avgFCT = []
for i in range(int(episode)):
    episode_DMR.append(DMR[step * i:step + step * i].sum() / step)
    episode_FCT.append(FCT[step * i:step + step * i].sum() / step)
    episode_EF_avgFCT.append(EF_avgFCT[step * i:step + step * i].sum() / step)
    episode_EF_FCT.append(EF_FCT[step * i:step + step * i].sum() / step)

plt.figure(1)
print episode_DMR
plt.xlabel('episode')
plt.ylabel('episode_DMR')
plt.plot(episode_DMR)

plt.figure(2)
plt.xlabel('episode')
plt.ylabel('episode_FCT')
plt.plot(episode_FCT)

plt.figure(3)
plt.xlabel('episode')
plt.ylabel('average EF_FCT')
plt.plot(episode_EF_avgFCT)
plt.show()

data_test = pd.read_csv('test_perform.csv', header=None)
perform = ['Size', 'FCT', 'Tput', 'EF_FCT', 'EF_num', 'EF_avgFCT', 'DMR']
data_test.columns = perform
# print data

DMR = data_test['DMR']
FCT = data_test['FCT']
EF_FCT = data_test['EF_FCT']
EF_avgFCT = data_test['EF_avgFCT']
DMR_sum = data_test['DMR'].sum()
FCT_sum = data_test['FCT'].sum()
EF_avgFCT_sum = data_test['EF_avgFCT'].sum()
EF_FCT_sum = data_test['EF_FCT'].sum()

Tput = data_test['Tput']
Tput_sum = data_test['Tput'].sum()

episode = data_test.shape[0] / step
print('episode:', episode, 'step:', step)
episode_DMR_test = []
episode_FCT_test = []
episode_EF_avgFCT_test = []
episode_EF_FCT_test = []

print('avg_DMR', DMR_sum / (episode * step) * 100)
print('avg_FCT', FCT_sum / (episode * step) * 1000)
print('avg_EF_FCT', EF_FCT_sum / (episode * step) * 1000)
print('EF_avgFCT', EF_avgFCT_sum / (episode * step) * 1000)
print('1/Tput', 1 / (Tput_sum / (episode * step)))

# for i in range(int(episode)):
#     episode_DMR_test.append(DMR[step * i:step + step * i].sum() / step)
# plt.plot(episode_DMR_test)
# plt.show()
