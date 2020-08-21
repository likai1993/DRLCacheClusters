import sys, os, random
import numpy as np
import pandas as pd

class DataLoader(object):
    def __init__(self):
        self.requests = []
        self.operations = []

    def get_requests(self):
        pass
    def get_operations(self):
        pass

    # map the disk address to table ID
    def get_tableIDs(self):
        table_range = [100, 500, 1500, 3000, 5000, 8000, 15000]
        tableIDs = []
        for request in self.requests:
            for i in range(len(table_range)):
                if int(request) < table_range[i]:
                    tableIDs.append(i)
                    break
        #print("get_tableIDs", len(self.requests), len(tableIDs))
        return tableIDs


class DataLoaderPintos(DataLoader):
    def __init__(self, progs, boot=False):
        super(DataLoaderPintos, self).__init__()

        if isinstance(progs, str): progs = [progs]
        for prog in progs:
            df = pd.read_csv(prog, header=0)
            if not boot: df = df.loc[df['boot/exec'] == 1, :]
            self.requests += list(df['blocksector'])
            self.operations += list(df['read/write'])

    def get_requests(self):
        return self.requests

    def get_operations(self):
        return self.operations

class DataLoaderMix(DataLoader):
    def __init__(self, progs, num_of_peroids, boot=False):
        super(DataLoaderMix, self).__init__()

        if isinstance(progs, str): progs = [progs]
        traces = {}
        operations = {}
        for prog in progs:
            traces[prog] = []
            operations[prog] = []
            for peroid in range(num_of_peroids):
                trace = prog.split("peroid")[0] + "peroid_" + str(peroid)
                df = pd.read_csv(trace, header=0)
                if not boot: df = df.loc[df['boot/exec'] == 1, :]
                trace = list(df['blocksector'])
                operation = list(df['read/write'])
                traces[prog] += trace
                operations[prog] += operation

        # mixing
        pivot = 0
        step = 10
        trace1 = traces[progs[0]]
        trace2 = traces[progs[1]]
        operations1 = operations[progs[0]]
        operations2 = operations[progs[1]]

        while(pivot +step < len(trace1) and pivot + step < len(trace2)):
            self.requests += trace1[pivot:pivot+10] 
            self.requests += trace2[pivot:pivot+10] 
            self.operations += operations1[pivot:pivot+10] 
            self.operations += operations2[pivot:pivot+10] 
            pivot += 10

        if pivot < len(trace1):
            self.requests += trace1[pivot:] 
            self.operations += operations1[pivot:] 
        if pivot < len(trace2):
            self.requests += trace2[pivot:] 
            self.operations += operations2[pivot:] 

        print(self.requests[0:2*step])

    def get_requests(self):
        return self.requests

    def get_operations(self):
        return self.operations

class DataLoaderZipf(DataLoader):
    def __init__(self, num_files, num_samples, param, num_progs=1, operation='random'):
        super(DataLoaderZipf, self).__init__()

        for i in range(num_progs):
            files = np.arange(num_files)
            # Random ranks. Note that it starts from 1.
            ranks = np.random.permutation(files) + 1
            # Distribution
            pdf = 1 / np.power(ranks, param)
            pdf /= np.sum(pdf)
            # Draw samples
            self.requests += np.random.choice(files, size=num_samples, p=pdf).tolist()
            if operation == 'random':
                self.operations += np.random.choice([0, 1], size=num_samples).tolist()
            else:
                self.operations += np.full(num_samples, int(operation)).tolist()

    def get_requests(self):
        return self.requests

    def get_operations(self):
        return self.operations
