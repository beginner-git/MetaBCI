import torch
import numpy as np
import multiprocessing
from queue import Empty

from metabci.brainflow.workers import ProcessWorker
from metabci.brainda.algorithms.deep_learning.models import CosCNN


class PredictionWorker(ProcessWorker):
    """
    一个专用的进程工作器，用于在独立进程中运行模型预测。
    它接收数据块，执行推理，并通过队列将结果发送回去。
    """

    def __init__(self, model_state: dict, model_config: dict, result_queue: multiprocessing.Queue, **kwargs):
        super().__init__(**kwargs)
        self._model_state = model_state
        self._model_config = model_config
        self.result_queue = result_queue
        self.model = None
        self.device = None

    def pre(self):
        """
        在主循环之前执行。
        负责初始化模型并将其移动到适当的设备上。
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = CosCNN(
            input_length=self._model_config['input_length'],
            in_channels=1,
            num_classes=self._model_config['num_classes'],
            filter_length=self._model_config['filter_length'],
            num_filters_list=self._model_config['num_filters_list']
        )
        self.model.load_state_dict(self._model_state)
        self.model.to(self.device)
        self.model.eval()
        print(f"PredictionWorker (PID: {self.pid}) initialized on {self.device}.")

    def consume(self, data: np.ndarray):
        """
        消费一个数据块，运行预测，并将结果放入结果队列。
        """
        try:
            input_tensor = torch.from_numpy(data).float()

            # ** 新增 **：处理数据填充和截断，以匹配模型输入
            expected_length = self._model_config.get('input_length')
            if expected_length and len(input_tensor) != expected_length:
                if len(input_tensor) > expected_length:
                    input_tensor = input_tensor[:expected_length]
                else:
                    padding = torch.zeros(expected_length - len(input_tensor))
                    input_tensor = torch.cat((input_tensor, padding))

            # 为模型准备张量形状：(batch_size, in_channels, sequence_length)
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
            input_tensor = input_tensor.to(self.device)

            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)

                result = {
                    "prediction": predicted_class.item(),
                    "confidence": confidence.item()
                }
                self.result_queue.put(result)

        except Exception as e:
            error_message = {"error": str(e)}
            self.result_queue.put(error_message)

    def post(self):
        """
        主循环结束后执行。清理资源。
        """
        self.model = None
        self.device = None
        print(f"PredictionWorker (PID: {self.pid}) finished.")
