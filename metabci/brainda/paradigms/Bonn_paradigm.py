# bonn_paradigm.py (终极决定版)

import mne
import numpy as np
import pandas as pd
from mne.utils import verbose

from metabci.brainda.utils import pick_channels
from metabci.brainda.paradigms.base import BaseParadigm


class BonnEEGParadigm(BaseParadigm):
    def __init__(self, events, tmin=0, tmax=None):
        super().__init__(events=events)
        self.tmin = tmin
        self.tmax = tmax

    def is_valid(self, dataset) -> bool:
        return True

    @verbose
    def _get_single_subject_data(self, dataset, subject_id, verbose=None):
        if not self.is_valid(dataset):
            raise TypeError(f"Dataset {dataset.dataset_code} is not valid for this paradigm.")

        used_events, used_intervals = self._map_events_intervals(dataset)
        Xs, ys, metas = {}, {}, {}

        # --------------------------- 终极修正：绕过“污染”数据的 get_data 方法 ---------------------------
        # 旧代码 (会调用父类方法，导致数据类型被意外改变):
        # data_from_dataset = dataset.get_data([subject_id])
        # 新代码 (直接调用我们自己写的、最纯净的数据提供方法):
        data_from_dataset = dataset._get_single_subject_data(subject_id)
        # ---------------------------------------------------------------------------------------------

        for subject, sessions in data_from_dataset.items():
            for session, runs in sessions.items():
                # 现在，raw_components 就是我们自己定义的那个纯净的元组，不再是字典
                for run, raw_components in runs.items():
                    # a. 直接解包这个100%可靠的元组，不再需要任何拆包操作
                    numpy_data, info, annotations = raw_components

                    # b. 组装成一个功能完整的MNE Raw对象
                    raw = mne.io.RawArray(numpy_data, info, verbose=False)
                    raw.set_annotations(annotations, verbose=False)

                    # 后续所有代码现在都可以正常工作
                    channels = self.select_channels if self.select_channels else dataset.channels
                    picks = pick_channels(raw.ch_names, channels, ordered=True)
                    events, _ = mne.events_from_annotations(raw, event_id=used_events)

                    tmin_val = self.tmin if self.tmin is not None else used_intervals[list(used_events.keys())[0]][0]
                    tmax_val = self.tmax if self.tmax is not None else used_intervals[list(used_events.keys())[0]][
                                                                           1] - 1.0 / raw.info['sfreq']

                    epochs = mne.Epochs(
                        raw, events,
                        event_id=used_events,
                        tmin=tmin_val, tmax=tmax_val,
                        picks=picks, proj=False, baseline=None, preload=True,
                        verbose=False
                    )

                    if len(epochs) == 0: continue
                    if self.srate: epochs = epochs.resample(self.srate, verbose=False)

                    X_epoched = epochs.get_data() * 1e6
                    y_epoched = epochs.events[:, -1]

                    for event_name in epochs.event_id.keys():
                        event_indices = np.where(y_epoched == used_events[event_name])[0]
                        if len(event_indices) == 0: continue

                        meta_df = pd.DataFrame({
                            "subject": subject, "session": session, "run": run,
                            "event": event_name, "trial_id": np.arange(len(event_indices)),
                            "dataset": dataset.dataset_code
                        })

                        if event_name not in Xs:
                            Xs[event_name], ys[event_name], metas[event_name] = [], [], []

                        Xs[event_name].append(X_epoched[event_indices])
                        ys[event_name].append(y_epoched[event_indices])
                        metas[event_name].append(meta_df)

        for event_name in Xs.keys():
            if Xs[event_name]:
                Xs[event_name] = np.concatenate(Xs[event_name], axis=0)
                ys[event_name] = np.concatenate(ys[event_name], axis=0)
                metas[event_name] = pd.concat(metas[event_name], axis=0, ignore_index=True)

        return Xs, ys, metas
