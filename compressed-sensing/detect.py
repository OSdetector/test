import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm


from utils_detect import data_process
from utils_detect.metrics import sliding_anomaly_predict
from algorithm.cluster import cluster
from algorithm.lesinn import online_lesinn
from algorithm.sampling.localized_sample import localized_sample
from algorithm.cvxpy import reconstruct
from cvxpy.error import SolverError

# some upper limit
max_seed = 10 ** 9 + 7


def anomaly_score_example(source: np.array, reconstructed: np.array):
    """
    Calculate anomaly score
    :param source: original data
    :param reconstructed: reconstructed data
    :return:
    """
    n, d = source.shape
    d_dis = np.zeros((d,))
    for i in range(d):
        dis = np.abs(source[:, i] - reconstructed[:, i])
        dis = dis - np.mean(dis)
        d_dis[i] = np.percentile(dis, 90)
    if d <= 2:
        return d / np.sum(1 / d_dis)
    topn = 1 / d_dis[np.argsort(d_dis)][-1 * 2:]
    return 2 / np.sum(topn)


class WindowReconstructProcess():
    """
    窗口重建工作进程
    """

    def __init__(
            self,
            data: np.array,
            cycle: int,
            latest_windows: int,
            sample_rate: float,
            scale: float,
            rho: float,
            sigma: float,
            random_state: int,
            retry_limit: int
    ):
        """
        :param data: 原始数据的拷贝
        :param cycle: 周期
        :param latest_windows: 计算采样价值指标时参考的最近历史周期数
        :param sample_score_method: 计算采样价值指标方法
        :param sample_rate: 采样率
        :param scale: 采样参数: 等距采样点扩充倍数
        :param rho: 采样参数: 中心采样概率
        :param sigma: 采样参数: 采样集中程度
        :param random_state: 随机数种子
        :param retry_limit: 每个窗口重试的上限
        """
        super().__init__()
        self.data = data
        self.cycle = cycle
        self.latest_windows = latest_windows
        self.sample_rate = sample_rate
        self.scale = scale
        self.rho = rho
        self.sigma = sigma
        self.random_state = random_state
        self.retry_limit = retry_limit

    def sample(self, x: np.array, m: int, score: np.array, random_state: int):
        """
        取得采样的数据
        :param x: kpi等距时间序列, shape=(n,d), n是行数, d是维度
        :param m: 采样个数
        :param score: 采样点置信度
        :param random_state: 采样随机种子
        :return: 采样序列数组X, X[i][0]是0~n-1的实数, 表示该采样点的时间点,
                X[i][1] 是shape=(k,)的数组, 表示该时间点各个维度kpi数据  0<i<m
                已经按X[i][0]升序排序
        """
        n, d = x.shape
        data_mat = np.mat(x)
        sample_matrix, timestamp = localized_sample(
            x=data_mat, m=m,
            score=score,
            scale=self.scale, rho=self.rho, sigma=self.sigma,
            random_state=random_state
        )
        # 采样中心对应的位置
        s = np.array(sample_matrix * data_mat)
        res = []
        for i in range(m):
            res.append((timestamp[i], s[i]))
        res.sort(key=lambda each: each[0])
        res = np.array(res)
        timestamp = np.array(res[:, 0]).astype(int)
        values = np.zeros((m, d))
        for i in range(m):
            values[i, :] = res[i][1]
        return timestamp, values

    def window_sample_reconstruct(
            self,
            data: np.array,
            groups: list,
            score: np.array,
            random_state: int
    ):
        """
        :param data: 原始数据
        :param groups: 分组
        :param score: 这个窗口的每一个点的采样可信度
        :param random_state: 随机种子
        :return: 重建数据, 重建尝试次数
        """
        # 数据量, 维度
        n, d = data.shape
        retry_count = 0
        sample_rate = self.sample_rate
        while True:
            try:
                timestamp, values = \
                    self.sample(
                        data,
                        int(np.round(sample_rate * n)),
                        score,
                        random_state
                    )
                rec = np.zeros(shape=(n, d))
                for i in range(len(groups)):
                    x_re = reconstruct(
                        n, len(groups[i]), timestamp,
                        values[:, groups[i]]
                    )
                    for j in range(len(groups[i])):
                        rec[:, groups[i][j]] = x_re[:, j]
                break
            except SolverError:
                if retry_count > self.retry_limit:
                    raise Exception(
                        'retry failed, please try higher sample rate or '
                        'window size'
                    )
                sample_rate += (1 - sample_rate) / 4
                retry_count += 1
                from sys import stderr
                stderr.write(
                    'WARNING: reconstruct failed, retry with higher '
                    'sample rate %f, retry times remain %d\n'
                    % (
                        sample_rate, self.retry_limit - retry_count)
                )
        return rec, retry_count


def detect(data_in, data_out, start):
    config = 'detector-config.yml'
    with open(config, 'r', encoding='utf8') as file:
        config_dict = yaml.load(file, Loader=yaml.Loader)
    data = pd.read_csv(data_in, header=1)
    # DEBUG
    # Chop down data size
    data = data.iloc[start:data.shape[0], 2:data.shape[1]]

    n, d = data.shape

    # Normalize each dimension
    data = data.values
    for i in range(d):
        data[:, i] = data_process.normalization(data[:, i])

    # 采样时参考的历史窗口数
    latest_windows = config_dict['detector_arguments']['latest_windows']
    # 采样率
    sample_rate = config_dict['detector_arguments']['sample_rate']
    rho = config_dict['detector_arguments']['rho']
    sigma = config_dict['detector_arguments']['sigma']
    scale = config_dict['detector_arguments']['scale']
    retry_limit = config_dict['detector_arguments']['retry_limit']
    random_state = config_dict['global']['random_state']

    # Get clustered group
    cluster_threshold = config_dict['detector_arguments']['cluster_threshold']
    windows_per_cycle = config_dict['data']['rec_windows_per_cycle']
    window = config_dict['data']['reconstruct']['window']
    stride = config_dict['data']['reconstruct']['stride']
    cycle = window * windows_per_cycle
    cycle_groups = []
    group_index = 0
    # 周期开始的index
    cb = 0
    while cb < n:
        # 周期结束的index
        ce = min(n, cb + cycle)  # 一周期数据为data[cb, ce)
        # 初始化追加列表引用
        if group_index == 0:
            # 没有历史数据
            # 分组默认每个kpi一组
            init_group = []
            for i in range(d):
                init_group.append([i])
            cycle_groups.append(init_group)
        else:
            cycle_groups.append(cluster(data[cb:ce], cluster_threshold))
        group_index += 1
        cb += cycle

    # 采样 & 重建
    process = WindowReconstructProcess(
        data=data,
        cycle=cycle,
        latest_windows=latest_windows,
        sample_rate=sample_rate,
        scale=scale, rho=rho, sigma=sigma,
        random_state=random_state,
        retry_limit=retry_limit
    )
    # 重建的数据
    reconstructed = np.zeros((n, d))
    reconstructing_weight = np.zeros((n,))
    needed_weight = np.zeros((n,))
    total_retries = 0
    win_l = 0
    win_r = 0
    pbar = tqdm(total=n)
    while win_r < n:
        win_r = min(n, win_l + window)
        group = cycle_groups[win_l // cycle]
        needed_weight[win_l:win_r] += 1

        # 采样概率
        hb = max(0, win_l - latest_windows)
        latest = data[hb:win_l]
        window_data = data[win_l:win_r]
        sample_score = online_lesinn(window_data, latest)
        rec_window, retries = \
            process.window_sample_reconstruct(
                data=window_data,
                groups=group,
                score=sample_score,
                random_state=random_state * win_l * win_r % max_seed
            )
        total_retries += retries
        for index in range(rec_window.shape[0]):
            w = index + win_l
            weight = reconstructing_weight[w]
            reconstructed[w, :] = \
                (reconstructed[w, :] * weight +
                 rec_window[index]) / (weight + 1)
        reconstructing_weight[win_l:win_r] += 1
        win_l += stride
        pbar.update(stride)

    pbar.close()

    # 预测
    # 异常得分
    anomaly_score = np.zeros((n,))
    # 表示当时某个位置上被已重建窗口的数量
    anomaly_score_weight = np.zeros((n,))
    # 窗口左端点索引
    wb = 0
    while True:
        we = min(n, wb + window)
        # 窗口右端点索引 窗口数据[wb, we)
        score = anomaly_score_example(data[wb:we], reconstructed[wb:we])
        for i in range(we - wb):
            w = i + wb
            weight = anomaly_score_weight[w]
            anomaly_score[w] = \
                (anomaly_score[w] * weight + score) / (weight + 1)
        anomaly_score_weight[wb:we] += 1
        if we >= n:
            break
        wb += stride

    # 接下来使用EVT等方式确定阈值，并做出检测
    predict = sliding_anomaly_predict(anomaly_score)

    np.savetxt(data_out, predict, delimiter=",")

    print("Done")
