import numpy as np
import json
import os
from typing import Tuple, Dict
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder


def stratified_k_fold(labels, k):
    """
    将标签集随机等分为 k 个 fold，保证每个 fold 中各个标签的数量尽可能相等

    参数:
    labels: 标签列表，例如 [0, 1, 2, 3, 0, 1, 2...]
    k: fold 的数量

    返回:
    fold_indices: 每个样本对应的 fold 索引（0 到 k-1）
    """
    unique_labels = np.unique(labels)
    label_indices = {}

    for label in unique_labels:
        label_indices[label] = np.where(np.array(labels) == label)[0]
        np.random.shuffle(label_indices[label])

    fold_indices = np.zeros(len(labels), dtype=int)

    for label, indices in label_indices.items():
        fold_sizes = np.ones(k, dtype=int) * (len(indices) // k)
        remainder = len(indices) % k
        if remainder > 0:
            fold_sizes[:remainder] += 1

        start_idx = 0
        for fold_idx in range(k):
            end_idx = start_idx + fold_sizes[fold_idx]
            fold_indices[indices[start_idx:end_idx]] = fold_idx
            start_idx = end_idx

    return fold_indices


class DataManager:
    """处理数据加载和预处理的类"""

    def __init__(self, sig_path: str, divide: int, data_export_dir: str = None):
        self.sig_path = sig_path  # 可以为 None
        self.divide = divide
        if data_export_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            self.data_export_dir = os.path.join(parent_dir, 'data')
        else:
            self.data_export_dir = data_export_dir
        os.makedirs(self.data_export_dir, exist_ok=True)

        # 确保目录存在
        os.makedirs(self.data_export_dir, exist_ok=True)

    def export_cv_indices_and_metadata(self, cv_indices: np.ndarray, num_classes: int) -> Dict[str, str]:
        """
        导出 cv_indices、num_classes 和 num_folds 到 JSON 文件

        参数:
        cv_indices: 交叉验证索引
        num_classes: 类别数量

        返回:
        Dict[str, str]: 包含导出文件路径的字典
        """
        file_paths = {}

        # 导出 cv_indices
        cv_indices_path = os.path.join(self.data_export_dir, 'cv_indices.json')
        with open(cv_indices_path, 'w') as f:
            json.dump(cv_indices.tolist(), f)
        file_paths['cv_indices'] = cv_indices_path

        # 导出 num_classes
        metadata_path = os.path.join(self.data_export_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump({'num_classes': int(num_classes)}, f)
        file_paths['metadata'] = metadata_path

        # 导出 divide 作为 num_folds
        num_folds_path = os.path.join(self.data_export_dir, 'num_folds.json')
        with open(num_folds_path, 'w') as f:
            json.dump({'num_folds': self.divide}, f)
        file_paths['num_folds'] = num_folds_path

        print(f"cv_indices、num_classes 和 num_folds 已导出到 {self.data_export_dir}")
        return file_paths

    def load_data_from_json(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        从 JSON 文件加载 sigData 和 labelData
        """
        sig_data_path = os.path.join(self.data_export_dir, 'sigData.json')
        label_data_path = os.path.join(self.data_export_dir, 'labelData.json')

        if not os.path.exists(sig_data_path) or not os.path.exists(label_data_path):
            raise FileNotFoundError(f"在 {self.data_export_dir} 中未找到所需的 JSON 文件 (sigData.json 或 labelData.json)")

        with open(sig_data_path, 'r') as f:
            sigData = np.array(json.load(f))

        with open(label_data_path, 'r') as f:
            labelData = np.array(json.load(f))

        print(f"sigData 和 labelData 已从 {self.data_export_dir} 加载")
        return sigData, labelData

    def load_and_preprocess_data(self, sig_data_path=None, label_data_path=None):
        """
        Load and preprocess signal data from JSON files, or raise an error if files are missing.

        Args:
            sig_data_path (str, optional): Path to signal data JSON file. Defaults to None.
            label_data_path (str, optional): Path to label data JSON file. Defaults to None.
        """
        # Use provided paths or default paths
        sig_json_path = sig_data_path or os.path.join(self.data_export_dir, "sigData.json")
        label_json_path = label_data_path or os.path.join(self.data_export_dir, "labelData.json")

        try:
            # Load data from JSON files
            with open(sig_json_path, 'r') as f:
                sigData = np.array(json.load(f))
            with open(label_json_path, 'r') as f:
                labelData = np.array(json.load(f))
        except FileNotFoundError:
            if self.sig_path is None:
                raise FileNotFoundError(
                    f"JSON files not found at {sig_json_path} and {label_json_path}, "
                    "and no original .mat file path (sig_path) was provided."
                )
            # If sig_path is provided, load from .mat file as a fallback (optional logic)
            print("JSON files not found. Loading from original .mat file...")
            data_sig = loadmat(self.sig_path)
            SIG = data_sig['SIG']
            indices_B = np.arange(100, 200)
            indices_D_E = np.arange(300, 500)
            indices = np.concatenate((indices_B, indices_D_E))
            sigData = SIG[indices, :-2]
            labelData = SIG[indices, -1].flatten()
            sigData = np.expand_dims(sigData, axis=1)
            sigData = sigData - 0.5
            # Optionally save to JSON for future use
            with open(sig_json_path, 'w') as f:
                json.dump(sigData.tolist(), f)
            with open(label_json_path, 'w') as f:
                json.dump(labelData.tolist(), f)

        # Additional preprocessing
        cv_indices = stratified_k_fold(labelData, self.divide)  # Assume this function exists
        cv_indices = cv_indices + 1

        le = LabelEncoder()
        labelData = le.fit_transform(labelData)
        num_classes = len(le.classes_)

        return sigData, labelData, cv_indices, num_classes

    def export_to_json(self, sigData, labelData):
        """
        将数据导出到 JSON 文件
        """
        sig_data_path = os.path.join(self.data_export_dir, 'sigData.json')
        label_data_path = os.path.join(self.data_export_dir, 'labelData.json')
        with open(sig_data_path, 'w') as f:
            json.dump(sigData.tolist(), f)
        with open(label_data_path, 'w') as f:
            json.dump(labelData.tolist(), f)
        print(f"sigData 和 labelData 已导出到 {self.data_export_dir}")

    @staticmethod
    def get_train_val_idx(y_train_val: np.ndarray, coeff: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
        """将数据分为训练集和验证集"""
        class_labels = np.unique(y_train_val)
        itrain = []
        ival = []

        for cls in class_labels:
            cur_class_idx = np.where(y_train_val == cls)[0]
            n_train = int(round(coeff * len(cur_class_idx)))
            itrain.extend(cur_class_idx[:n_train])
            ival.extend(cur_class_idx[n_train:])

        return np.array(itrain), np.array(ival)