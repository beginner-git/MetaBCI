import mne
import numpy as np
from pathlib import Path
from typing import List, Optional, Union

from metabci.brainda.datasets.base import BaseDataset


class BonnEEGDataset(BaseDataset):
    """
    Bonn EEG Dataset.
    """

    def __init__(self, path: str, subject_ids: List[int] = None):
        if subject_ids is None:
            processed_subjects = [f"{i:03d}" for i in range(1, 101)]
        else:
            processed_subjects = [f"{i:03d}" for i in subject_ids]

        srate = 173.61
        trial_len_in_samples = 4096
        trial_duration = trial_len_in_samples / srate
        default_interval = (0, trial_duration)

        events_with_interval = {
            'O': (1, default_interval),
            'F': (2, default_interval),
            'S': (3, default_interval)
        }

        super().__init__(
            subjects=processed_subjects,
            paradigm='MI',
            dataset_code='BonnEEG',
            events=events_with_interval,
            srate=srate,
            channels=['EEG1']
        )

        self.root = Path(path)
        self.trial_len_in_samples = trial_len_in_samples
        self.trial_len_in_seconds = trial_duration

    def get_data(self, subjects: Optional[List[Union[int, str]]] = None, verbose: Optional[bool] = None) -> dict:
        if subjects is None:
            return super().get_data(subjects=None, verbose=verbose)
        processed_subjects = [
            f"{s:03d}" if isinstance(s, int) else s for s in subjects
        ]
        return super().get_data(subjects=processed_subjects, verbose=verbose)

    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None):
        return [self.root]

    def _get_single_subject_data(self, subject: Union[str, int]) -> dict:
        """
        核心方法：为单个被试加载数据。
        """
        all_trials_data, event_names = [], []
        for event_name in self.events.keys():
            # 关键修正：使用 :03d 格式化，将整数 1 变成字符串 '001'
            file_path = self.root / event_name / f"{event_name}{int(subject):03d}.txt"
            if file_path.exists():
                all_trials_data.append(np.loadtxt(file_path, dtype=np.float64))
                event_names.append(event_name)

        if not all_trials_data:
            return {str(subject): {}}

        concatenated_data = np.concatenate(all_trials_data).reshape(1, -1)

        info = mne.create_info(ch_names=self.channels, sfreq=self.srate, ch_types=['eeg'])

        annotations = mne.Annotations(
            onset=[i * self.trial_len_in_seconds for i in range(len(event_names))],
            duration=[self.trial_len_in_seconds] * len(event_names),
            description=event_names
        )

        raw_components = (concatenated_data, info, annotations)

        return {str(subject): {'session_0': {'run_0': raw_components}}}
    # -----------------------------------------------------------------------------
