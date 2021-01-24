import csv
import pandas as pd

a = []
df = pd.DataFrame(a)
df.to_csv("./data/train_perform.csv", mode='w', header=False, index=False)
df.to_csv("./data/train_perform2.csv", mode='w', header=False, index=False)
df.to_csv("./data/test_rewards.csv", mode='w', header=False, index=False)
df.to_csv("./data/test_perform.csv", mode='w', header=False, index=False)
df.to_csv("./data/test_perform2.csv", mode='w', header=False, index=False)
df.to_csv("./data/flow.csv", mode='w', header=False, index=False)
