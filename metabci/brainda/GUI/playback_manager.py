import numpy as np
import time


class PlaybackManager:
    """
    Manages the state and data access for a time-accurate playback simulation.

    This class holds the complete historical data but only exposes the parts
    that should have "arrived" based on the elapsed time. This simulates a
    real-time data source that cannot see the future, while ensuring that the
    playback is perfectly synchronized with real clock time.
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
        """Starts the playback timer."""
        self.is_running = True
        self.start_time = time.time()

    def stop(self):
        """Stops the playback."""
        self.is_running = False

    def get_total_length(self):
        """Returns the total number of samples in the full data stream."""
        return self._total_data_length

    def get_full_data_for_final_plot(self):
        """After the animation finishes, returns the complete dataset for a final static plot."""
        return self._x_data, self._full_data_stream

    def get_data_for_current_time(self):
        """
        Calculates and returns the data slice corresponding to the current elapsed time.
        This is the core method for simulating real-time data arrival.
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
        """Checks if the playback has naturally concluded."""
        # When it is no longer running, and it has been started before, it means it has finished.
        return not self.is_running and self.start_time > 0