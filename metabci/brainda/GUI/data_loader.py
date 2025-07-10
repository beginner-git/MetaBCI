import json
import os
from typing import Dict, Tuple

import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder


def main(export_to_json: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载并预处理信号数据，并可选择导出为JSON

    参数:
    export_to_json: 是否将数据导出为JSON文件

    返回:
    Tuple[np.ndarray, np.ndarray]: 包含信号数据（形状为 (num_data, 1, data_length)）和标签数据的元组
    """
    # 定义 sig_path 为上一层文件夹的 data 文件夹下的 SIG.mat 文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), "data")
    sig_path = os.path.join(data_dir, "SIG.mat")

    print("Processing data from source file...")
    data_sig = loadmat(sig_path)
    SIG = data_sig['SIG']

    # 提取指定类别
    indices_B = np.arange(100, 200)
    indices_D_E = np.arange(300, 500)
    indices = np.concatenate((indices_B, indices_D_E))

    sigData = SIG[indices, :-2]
    labelData = SIG[indices, -1].flatten()

    # 修改 sigData 的形状，增加一个通道维度
    sigData = np.expand_dims(sigData, axis=1)  # (num_data, data_length) -> (num_data, 1, data_length)

    # 数据居中处理
    sigData = sigData - 0.5

    # 标签编码
    le = LabelEncoder()
    labelData = le.fit_transform(labelData)

    # 导出到JSON（如果需要）
    if export_to_json:
        export_data_to_json(sigData, labelData)

    print(f"Shape of sigData: {sigData.shape}") # 添加打印形状确认
    print(f"Shape of labelData: {labelData.shape}")

    return sigData, labelData


def export_data_to_json(sigData: np.ndarray, labelData: np.ndarray) -> Dict[str, str]:
    """
    将数据导出为JSON文件

    参数:
    sigData: 信号数据 (形状应为 (num_data, 1, data_length))
    labelData: 标签数据

    返回:
    Dict[str, str]: 包含各JSON文件路径的字典
    """
    file_paths = {}

    # 定义导出目录为上一级文件夹下的 'data' 文件夹
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_export_dir = os.path.join(os.path.dirname(current_dir), "data")

    # 确保目录存在
    os.makedirs(data_export_dir, exist_ok=True)

    # 导出 sigData
    sig_data_path = os.path.join(data_export_dir, 'sigData.json')
    with open(sig_data_path, 'w') as f:
        # tolist() 可以处理多维数组
        json.dump(sigData.tolist(), f)
    file_paths['sigData'] = sig_data_path

    # 导出 labelData
    label_data_path = os.path.join(data_export_dir, 'labelData.json')
    with open(label_data_path, 'w') as f:
        json.dump(labelData.tolist(), f)
    file_paths['labelData'] = label_data_path

    print(f"Data exported to {data_export_dir}")
    return file_paths

if __name__ == '__main__':
    main()
