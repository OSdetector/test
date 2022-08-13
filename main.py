import os
import sys
import time
import argparse
import multiprocessing as mp

sys.path.append(os.getcwd() + '/snoop')

import top_snoop

if __name__=='__main__':

    # 解析参数
    configure = top_snoop.parse_args()
    print(configure["detect_interval"])

    snoop_process = mp.Process(target=top_snoop.start, args=(configure, ))
    snoop_process.start()

    # 四个csv上一次检测到的最后一行
    last = [0, 0, 0, 0]
    while True:
        """
        csv in ./csv/
        outcome in ./outcome/
        """
        last[0] = len(open("csv/cpu.csv", 'r').readlines())
        last[1] = len(open("csv/mem.csv", 'r').readlines())
        last[2] = len(open("csv/net.csv", 'r').readlines())
        last[3] = len(open("csv/syscall.csv", 'r').readlines())
        print(last)
        time.sleep(configure['detect_interval'])
