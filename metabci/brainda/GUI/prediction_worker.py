import torch
import numpy as np
import multiprocessing
from queue import Empty

from metabci.brainflow.workers import ProcessWorker
from metabci.brainda.algorithms.deep_learning.models import CosCNN


class PredictionWorker(ProcessWorker):
    """
    A dedicated process worker for running model predictions in a separate process.
    It receives data chunks, performs inference, and sends the results back through a queue.
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
        Executes before the main loop.
        Responsible for initializing the model and moving it to the appropriate device.
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
        Consumes a data chunk, runs a prediction, and places the result into the result queue.
        """
        try:
            input_tensor = torch.from_numpy(data).float()

            # ** NEW **: Handle data padding and truncation to match model input.
            expected_length = self._model_config.get('input_length')
            if expected_length and len(input_tensor) != expected_length:
                if len(input_tensor) > expected_length:
                    input_tensor = input_tensor[:expected_length]
                else:
                    padding = torch.zeros(expected_length - len(input_tensor))
                    input_tensor = torch.cat((input_tensor, padding))

            # Prepare tensor shape for the model: (batch_size, in_channels, sequence_length)
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
        Executes after the main loop finishes. Cleans up resources.
        """
        self.model = None
        self.device = None
        print(f"PredictionWorker (PID: {self.pid}) finished.")