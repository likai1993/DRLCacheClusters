#! /usr/bin/python3
import os, sys
file1 = sys.argv[1]
file2 = sys.argv[2]
epoch1 = int(sys.argv[3])
epoch2 = int(sys.argv[4])

output = file1.split("/")[-1] + file2.split("/")[-1]
with open (output, "w") as out:
    with open(file1, "r") as f1:
        f1s = f1.readlines()
        out.write(f1s[0])
        f1s = f1s[1:]
        with open (file2, "r") as f2:
            f2s = f2.readlines()[1:]
            pivot1=0
            pivot2=0
            while(pivot1<len(f1s) and pivot2 < len(f2s)):
                for item in f1s[pivot1:pivot1+epoch1]:
                    out.write(item)
                for item in f2s[pivot2:pivot2+epoch2]:
                    out.write(item)
                pivot1+=epoch1
                pivot2+=epoch2

            if pivot1 < len(f1s):
                for item in f1s[pivot1:]:
                    out.write(item)
            if pivot2 < len(f2s):
                for item in f2s[pivot2:]:
                    out.write(item)
