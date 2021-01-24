# -*- coding: utf-8 -*-

import pandas as pd

a = []
df = pd.DataFrame(a)
df.to_csv("./data/episode_data.csv", mode='w', header=False, index=False)
