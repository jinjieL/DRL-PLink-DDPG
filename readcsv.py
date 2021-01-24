import pandas as pd
import time

# data = pd.read_csv('1.csv', header=None, engine='python')

while True:
    try:
        data = pd.read_csv('1.csv', header=None, engine='python')
    except:
        print 'error'
        continue
t1 = time.time()
file_time = open('time.txt', mode='w')
reward = 1
t=time.time()-t1
file_time.write(str(reward) +'\n' + str(t))