# import threading
# import time
# from queue import Queue
#
# class DataStreamer(threading.Thread):
#     """
#     一个通过队列以指定频率推送数据点的线程。
#
#     该类实现了一种“推”模型的数据流。虽然目前的绘图实现
#     为了保证时间精确性而采用了基于时钟的“拉”模型，但该类
#     作为一个独立的组件被保留下来，用于其他可能的模拟场景。
#     """
#     def __init__(self, data: list, data_queue: Queue, frequency: float, stop_event: threading.Event):
#         super().__init__(daemon=True)
#         self.data = data
#         self.data_queue = data_queue
#         self.sleep_interval = 1.0 / frequency if frequency > 0 else 0
#         self.stop_event = stop_event
#
#     def run(self):
#         """
#         逐个将数据点放入队列，并伴有延迟。
#         """
#         for point in self.data:
#             if self.stop_event.is_set():
#                 break  # 如果绘图窗口关闭或停止，则退出
#
#             self.data_queue.put(point)
#             time.sleep(self.sleep_interval)
#
#         # 通过放入哨兵值(None)来标志数据流的结束
#         if not self.stop_event.is_set():
#             self.data_queue.put(None)
#
