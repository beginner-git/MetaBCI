import tkinter as tk
from tkinter import ttk
import numpy as np
from queue import Empty
import torch
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation


class RealtimePlotWindow:
    """
    管理一个高性能、时间精确的数据回放窗口。
    它从一个外部的 'playback_manager' 获取数据，确保自身无法访问“未来”的数据。
    当需要预测时，它会将数据发送到一个独立的后台工作进程。
    """

    def __init__(self, parent, playback_manager, channel_index,
                 true_label=None, next_callback=None,
                 prediction_worker=None, result_queue=None):

        self.parent = parent
        self.playback_manager = playback_manager
        self.channel_index = channel_index
        self.true_label = true_label
        self.next_callback = next_callback

        # 存储工作进程和结果队列
        self.worker = prediction_worker
        self.queue = result_queue
        self.is_waiting_for_prediction = False

        self.anim = None
        self.is_running = False

        self.window = tk.Toplevel(parent)
        title_sampling_rate = self.playback_manager._sampling_rate
        self.window.title(f"Playback Plot - Channel {self.channel_index} @ {title_sampling_rate:.1f} Hz")
        self.window.geometry("800x450")

        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)

        initial_title = f"Playback: Channel {self.channel_index}"
        if self.true_label is not None:
            initial_title += f" (True Label: {self.true_label})"
        self.ax.set_title(initial_title)
        self.ax.set_xlabel("Time (samples)")
        self.ax.set_ylabel("Amplitude")

        self.total_data_length = self.playback_manager.get_total_length()
        self.line, = self.ax.plot([], [], lw=1.5, label=f"Channel {self.channel_index}")

        self.ax.set_xlim(0, self.total_data_length)
        self.ax.set_ylim(-1, 1)
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.ax.legend()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        button_frame = ttk.Frame(self.window)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        self.stop_button = ttk.Button(button_frame, text="■ Stop", command=self.stop_plotting)
        self.stop_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        if self.next_callback:
            self.next_button = ttk.Button(button_frame, text="Next Data ▶▶", command=self.show_next, state='disabled')
            self.next_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        else:
            self.next_button = None

        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        self.window.after(100, self.start_animation)

    def start_animation(self):
        if self.is_running:
            return
        self.is_running = True
        self.playback_manager.start()

        self.anim = FuncAnimation(
            self.fig,
            self._update_frame,
            init_func=self._init_animation,
            interval=25,
            blit=True,
            repeat=False
        )
        self.canvas.draw_idle()

    def _init_animation(self):
        self.line.set_data([], [])
        return (self.line,)

    def _update_frame(self, frame_num):
        x_slice, y_slice = self.playback_manager.get_data_for_current_time()

        if self.playback_manager.has_finished():
            if self.is_running:
                self.is_running = False
                self.window.after(50, lambda: self.finalize_plot(run_prediction=True))

        if y_slice.size > 0:
            min_val, max_val = np.min(y_slice), np.max(y_slice)
            current_ax_min, current_ax_max = self.ax.get_ylim()

            if min_val < current_ax_min or max_val > current_ax_max:
                padding = (max_val - min_val) * 0.1 or 1.0
                self.ax.set_ylim(min_val - padding, max_val + padding)
                self.line.set_data(x_slice, y_slice)
                self.canvas.draw()
                return (self.line,)

        self.line.set_data(x_slice, y_slice)
        return (self.line,)

    def stop_plotting(self):
        if self.is_running:
            self.is_running = False
            self.playback_manager.stop()
            self.finalize_plot(run_prediction=False)

    def finalize_plot(self, run_prediction: bool):
        if self.anim and self.anim.event_source:
            self.anim.event_source.stop()

        final_x, final_y = self.playback_manager.get_full_data_for_final_plot()
        self.line.set_data(final_x, final_y)

        # 将预测任务交给后台进程
        if run_prediction and self.worker:
            _, data_for_prediction = self.playback_manager.get_full_data_for_final_plot()
            if data_for_prediction.size > 0:
                self.worker.put(data_for_prediction)
                self.is_waiting_for_prediction = True
                self._wait_for_prediction_and_update_ui()  # 开始轮询结果

        self.stop_button.config(state='disabled')
        if self.next_button:
            self.next_button.config(state='normal')

        self.canvas.draw_idle()

    def _wait_for_prediction_and_update_ui(self):
        """非阻塞地轮询队列以获取预测结果。"""
        if not self.is_waiting_for_prediction:
            return

        try:
            result = self.queue.get_nowait()
            self.is_waiting_for_prediction = False  # 收到结果，停止轮询

            if "error" in result:
                self.ax.set_title(f"Channel {self.channel_index} - Prediction Error")
            else:
                prediction = result['prediction']
                final_title = f"Channel {self.channel_index} - Predicted Class: {prediction}"
                if self.true_label is not None:
                    is_correct = (int(self.true_label) == prediction)
                    correctness_str = "Correct" if is_correct else "Incorrect"
                    final_title = f"True: {self.true_label}, Predicted: {prediction} ({correctness_str})"

                self.ax.set_title(final_title)
                self.change_bg_color(prediction)

            self.canvas.draw_idle()

        except Empty:
            # 队列为空，50毫秒后再次检查
            self.window.after(50, self._wait_for_prediction_and_update_ui)

    def show_next(self):
        if self.next_callback:
            self.next_callback()
        self.on_close()

    def change_bg_color(self, prediction_index):
        color_map = {0: ('lightgreen', 0.3), 1: ('lightyellow', 0.3), 2: ('lightcoral', 0.3)}
        color, alpha = color_map.get(prediction_index, ('lightgrey', 0.3))
        self.ax.set_facecolor(color)
        self.ax.patch.set_alpha(alpha)

    def on_close(self):
        if self.is_running:
            self.is_running = False
            self.playback_manager.stop()

        # 确保轮询循环停止
        self.is_waiting_for_prediction = False

        if self.anim and self.anim.event_source:
            self.anim.event_source.stop()

        self.window.destroy()
