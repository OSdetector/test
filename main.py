import os
import sys
import time
import argparse
import multiprocessing as mp

sys.path.append(os.getcwd() + '/snoop')
sys.path.append(os.getcwd() + '/compressed-sensing')

import top_snoop
import detect

if __name__=='__main__':

    # 解析参数
    configure = top_snoop.parse_args()
    print(configure["detect_interval"])

    snoop_process = mp.Process(target=top_snoop.start, args=(configure, ))
    snoop_process.start()

    # 上一次检测到的最后一行
    last = [0, 0, 0]
    while True:
        time.sleep(configure['detect_interval'])
        detect.detect("csv/cpu.csv", "outcome/cpu.csv", last[0])
        detect.detect("csv/mem.csv", "outcome/mem.csv", last[1])
        detect.detect("csv/net.csv", "outcome/net.csv", last[2])

        last[0] = len(open("csv/cpu.csv", 'r').readlines())
        last[1] = len(open("csv/mem.csv", 'r').readlines())
        last[2] = len(open("csv/net.csv", 'r').readlines())

