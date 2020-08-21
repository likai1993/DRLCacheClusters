#! /usr/bin/python3

import os, sys, time
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import random

if __name__ == "__main__":

    if len(sys.argv) != 6:
        print("Usage: %s <save_path> <num_resources> <num_samples> <zipf_param> <num_progs>" % sys.argv[0])
        exit(0)

    save_path = sys.argv[1]
    num_resources = int(sys.argv[2])
    num_samples = int(sys.argv[3])
    param = float(sys.argv[4])
    num_progs = int(sys.argv[5])

    df = None 
    for i in range(num_progs):
        files = np.arange(num_resources)
        # Random ranks. Note that it starts from 1.
        ranks = np.random.permutation(files) + 1

        # Distribution
        pdf = 1 / np.power(ranks, param)
        pdf /= np.sum(pdf)

        # Draw samples
        starts = np.random.choice(files, size=num_samples, p=pdf)
        operations = np.full_like(starts, 0)
        executions = np.full_like(starts, 1)

        requests1=[]
        requests2=[]
        requests3=[]
        for i in range(len(starts)):
            end = random.randrange(starts[i], num_resources)
            for j in range(starts[i], end):
                requests1.append(j) 
                requests2.append(operations[i])
                requests3.append(executions[i])
                print(j,operations[i],executions[i])

        tmp = pd.DataFrame({'blocksector': requests1, 'read/write': requests2, 'boot/exec': requests3})
        if df is None:
            df = tmp
        else:
            df = pd.concat((df, tmp), axis=0)
    df.to_csv(save_path, index=False, header=True)
