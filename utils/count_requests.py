#! /usr/bin/python3
import os, sys
import pandas as pd
from collections import Counter,OrderedDict # improve the counter speed in _elapsed_requests


file1 = sys.argv[1]
block_num = []
df = pd.read_csv(file1, header=0)
df = df.loc[df['boot/exec'] == 1, :]
trace = list(df['blocksector'])
C = Counter(trace)
SortedC = OrderedDict(C.most_common())
for key,value in SortedC.items():
    print(key,value)
