# OSdetector

OSdetector 是一个结合了 eBPF 和 LSTM、压缩感知等异常检测算法的智能操作系统异常检测程序，可以进行进程级别的数据采集和异常数据检测。

该项目整体使用 python 实现，包括 bcc 框架和 python 提供的各项数据处理工具，具体可参考安装说明。

> README 主要介绍安装和使用方法，项目具体实现细节请参考项目文档。

## 安装说明

### 1. 安装 bcc

安装依赖：

```
apt-get install -y build-essential git bison cmake flex  libedit-dev libllvm7 llvm-7-dev libclang-6.0-dev python zlib1g-dev libelf-dev python3-distutils libfl-dev
```

下载 release 包（git 仓库最新版 bcc 可能导致某些不兼容，因此选择稍低的 realease 版本）：

```
wget https://github.com/iovisor/bcc/releases/download/v0.24.0/bcc-src-with-submodule.tar.gz
tar -xzf bcc-src-with-submodule.tar.gz
```

编译项目：

```
mkdir bcc/build && cd bcc/build
cmake ..
make && make install
```

通过运行 `/usr/share/bcc/examples/hello_world.py` 检测是否安装成功。常见的报错可以在官方提供的<a href="https://github.com/iovisor/bcc/blob/master/FAQ.txt"> FAQ </a>中找到解决方案。

### 2. 安装其余依赖

```
pip install -r requirement.txt
```

## 使用说明

### 配置文件说明

项目整体配置：`config.json`, 可以参考目录中的 `config_template.json`。

```json
{
    "snoop_cpu": "None",
    "cpu_output_file": "cpu.csv",
    "snoop_mem": "None",
    "mem_output_file": "mem.csv",
    "probes":{
        "event_name":["./mem_example/main:func1"],
        "output_file":"tmp.csv",
        "spid":10000
    },
    "trace": {
        "tracee_name":["./test/search:search"],
        "output_file":"trace_output",
        "enter_msg_format":["'Enter search, n=%d' % (arg1)"],
        "return_msg_format":["Return search"]
        "spid":10000
    },
    "snoop_network": "bcc",
    "network_output_file": "net.csv",
    "snoop_syscall": "None",
    "syscall_output_file": "syscall.csv",
    "interval": 5,
    "trace_multiprocess":  true
    "show_all_threads": true

    "csvpath": "csv",
    "outcomepath": "outcome",
    "detect_interval": 60
}
```

其释义如下：

+ snoop_cpu: 进程CPU占用监控模块，可选项为("bcc", "stat", "top", null)
+ cpu_output_file: 进程CPU占用输出文件名
+ snoop_mem: 进程内存占用监控模块，可选项为("bcc", null)
+ mem_output_file: 进程内存占用输出文件名
+ probes: 用户态函数执行情况统计，包括内存占用变化，CPU时间分布
    + event_name: 检测的函数挂载点，bin:func
    + output_file: 输出文件
    + spid: 对某个特定线程进行监控，spid为线程id，填入null表示不启用
+ trace: 用户态函数执行跟踪
    + tracee_name: 检测的函数挂载点，bin:func
    + output_file: 输出文件
    + enter_msg_format: 进入函数时输出的消息，支持的参数为arg1-arg6，类型为数字或字符串
    + return_msg_format: 函数返回时输出的消息，支持的参数为retval
    + spid: 对某个特定线程进行监控，spid为线程id，填入null表示不启用
+ snoop_network: 进程网络流量监控模块，可选项为("bcc", null)
+ network_output_file: 进程流量监控输出文件名
+ snoop_syscall: 系统占用监控模块，可选项为("bcc", null)
+ syscall_output_file: 进程系统调用输出文件名
+ interval: 监控周期，单位秒
+ trace_multiprocess: 是否跟踪并监控进程产生的子进程（对系统调用监控输出会破坏原有顺序）
+ show_all_threads: 是否分别展示每个线程，设定为true输出文件中的PID属性改为SPID（线程号）

+ csvpath: 存储收集信息生成的 csv 文件的目录
+ outcomepath: 存储异常检测结果的目录
+ detect_interval: 对 csv 文件进行异常检测算法的时间间隔


算法参数配置：`compressed-sensing/detector-config.yml`

### 使用方法说明

```
python3 main.py -c './snoop_program' # 运行程序并进行数据收集和异常检测, 使用默认配置文件
python3 main.py -c './snoop_program' --configure_file ./config.json # 运行程序并进行数据收集和异常检测，使用指定配置文件
python3 main.py -p 12345  # Snoop the process with pid 12345 # 对目标进程进行数据收集和异常检测
```

结果可以在指定的 `outcomepath` 中查看
