import numpy as np
import time


class PlaybackManager:
    """
    管理时间精确的播放模拟的状态和数据访问。

    该类持有完整的历史数据，但只暴露那些根据已流逝时间
    理应“已经到达”的部分。这模拟了一个无法看到未来的实时数据源，
    同时确保了回放与真实时钟时间完美同步。
    """

    def __init__(self, full_data_stream, sampling_rate, chunk_size):
        self._full_data_stream = np.asarray(full_data_stream)
        self._sampling_rate = sampling_rate
        self._chunk_size = chunk_size
        self._total_data_length = len(self._full_data_stream)
        self._x_data = np.arange(self._total_data_length)

        self.start_time = 0.0
        self.is_running = False

    def start(self):
        """启动播放计时器。"""
        self.is_running = True
        self.start_time = time.time()

    def stop(self):
        """停止播放。"""
        self.is_running = False

    def get_total_length(self):
        """返回完整数据流的总样本数。"""
        return self._total_data_length

    def get_full_data_for_final_plot(self):
        """在动画结束后，返回完整的数据集用于最终的静态绘图。"""
        return self._x_data, self._full_data_stream

    def get_data_for_current_time(self):
        """
        计算并返回与当前流逝时间对应的数据切片。
        这是模拟实时数据到达的核心方法。
        """
        if not self.is_running:
            return np.array([]), np.array([])

        elapsed_seconds = time.time() - self.start_time

        num_points_ideal = int(elapsed_seconds * self._sampling_rate)

        num_chunks_to_show = num_points_ideal // self._chunk_size
        num_points_to_show = num_chunks_to_show * self._chunk_size

        if num_points_to_show >= self._total_data_length:
            num_points_to_show = self._total_data_length
            if self.is_running:
                self.is_running = False

        x_slice = self._x_data[:num_points_to_show]
        y_slice = self._full_data_stream[:num_points_to_show]

        return x_slice, y_slice

    def has_finished(self):
        """检查播放是否已经自然结束。"""
        # 当它不再运行，且曾经启动过，就意味着结束了
        return not self.is_running and self.start_time > 0
