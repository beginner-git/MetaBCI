import unittest
import os
import shutil
import json
import numpy as np
import pandas as pd

# 假设：请根据您的实际项目结构调整以下导入路径
# 测试需要能够导入您的 BonnEEGParadigm 和 BonnEEGDataset 类。
try:
    from metabci.brainda.paradigms.Bonn_paradigm import BonnEEGParadigm
    from metabci.brainda.datasets.bonn_eeg import BonnEEGDataset
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure the 'metabci' package is installed or its path is correctly configured.")
    # 如果无法导入，将无法运行测试，因此直接退出
    exit()


class TestBonnParadigmExport(unittest.TestCase):
    """
    BonnEEGParadigm 数据处理与导出功能的测试套件。

    本测试验证范式能否正确处理 Bonn 数据集（包括加载、排序、缩放），
    并将其导出为 GUI 应用所需的多个 JSON 文件。
    """

    @classmethod
    def setUpClass(cls):
        """
        设置测试环境。
        - 定义原始数据和测试输出的路径。
        - 为导出的文件创建一个临时的测试目录。
        """
        # 项目根目录，假设测试脚本位于根目录下的 'tests/' 文件夹中
        cls.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        # 数据集路径，与 export_eeg_data 中的定义保持一致
        cls.raw_data_path = os.path.join(cls.project_root, 'Dataset', 'Bonn_EEG')

        # 临时输出目录
        cls.output_path = os.path.join(os.path.dirname(__file__), 'temp_bonn_test_output')

        # 检查原始数据路径是否存在，如果不存在则跳过测试
        if not os.path.isdir(cls.raw_data_path):
            raise unittest.SkipTest(
                f"Bonn EEG dataset not found at the specified path: {cls.raw_data_path}. "
                "Please ensure the dataset is in 'MetaBCI/Dataset/Bonn_EEG/' relative to the project root."
            )

        # 为测试输出创建一个干净的目录
        if os.path.exists(cls.output_path):
            shutil.rmtree(cls.output_path)
        os.makedirs(cls.output_path)
        print(f"Created temporary directory for test outputs: {cls.output_path}")

    @classmethod
    def tearDownClass(cls):
        """测试结束后，清理测试环境，移除临时文件和目录。"""
        if os.path.exists(cls.output_path):
            shutil.rmtree(cls.output_path)
        print(f"\nCleaned up temporary directory: {cls.output_path}")

    def test_full_processing_and_sorted_export(self):
        """
        测试完整的数据处理、排序和 JSON 导出工作流。
        该测试模拟 `export_eeg_data` 脚本中的核心逻辑。
        """
        print("\nRunning Test: test_full_processing_and_sorted_export...")

        # 步骤 1: 实例化范式和数据集对象，与 export_eeg_data.py 保持一致
        events_with_labels = {'O': 0, 'F': 1, 'S': 2}
        all_subjects = list(range(1, 101))

        paradigm = BonnEEGParadigm(events=events_with_labels)
        dataset = BonnEEGDataset(path=self.raw_data_path)

        # 步骤 2: 运行核心的数据处理方法，并触发导出
        # 注意：此处添加 export_path 参数，以测试范式内部的导出逻辑
        X, y, meta = paradigm.get_data(
            dataset=dataset,
            subjects=all_subjects,
            return_concat=True,
            n_jobs=-1,
            export_path=self.output_path  # 关键：指定导出路径
        )

        # 步骤 3: 验证 get_data 方法返回的对象是否符合预期
        expected_trials = len(all_subjects) * len(events_with_labels)  # 100 * 3 = 300
        self.assertEqual(X.shape, (expected_trials, 1, 4096), "返回的 X (信号) 形状不正确")
        self.assertEqual(y.shape, (expected_trials,), "返回的 y (标签) 形状不正确")
        self.assertIsInstance(meta, pd.DataFrame, "返回的 meta 应为 pandas DataFrame")
        self.assertEqual(len(meta), expected_trials, "meta 中的试验数量不正确")
        print("OK: get_data 返回了有效的数据结构和匹配的维度。")

        # 步骤 4: 验证所有预期的文件是否都已创建
        expected_files = ['sigData.json', 'labelData.json', 'metadata.json']  # cv_indices.json 可能为可选
        for filename in expected_files:
            file_path = os.path.join(self.output_path, filename)
            self.assertTrue(os.path.exists(file_path), f"文件 {filename} 未被创建。")
        print(f"OK: 所有预期的 JSON 文件都已在 {self.output_path} 中创建。")

        # 步骤 5: 验证导出的 JSON 文件内容是否经过了正确的排序和缩放处理

        # --- 模拟 export_eeg_data.py 中的排序和缩放逻辑，以生成期望结果 ---
        meta_copy = meta.copy()
        meta_copy.sort_values(by=['event', 'subject'], inplace=True)
        sorted_indices = meta_copy.index

        expected_x_sorted_scaled = X[sorted_indices] / 1e6  # 应用缩放
        expected_y_sorted = y[sorted_indices]

        # --- 从文件中加载数据进行比较 ---
        with open(os.path.join(self.output_path, 'sigData.json'), 'r') as f:
            x_from_file = np.array(json.load(f))

        with open(os.path.join(self.output_path, 'labelData.json'), 'r') as f:
            y_from_file = np.array(json.load(f))

        # --- 执行比较 ---
        np.testing.assert_allclose(x_from_file, expected_x_sorted_scaled, rtol=1e-8,
                                   err_msg="sigData.json 的内容与预期（排序和缩放后）不符。")
        print("OK: sigData.json 内容经过了正确的排序和缩放。")

        np.testing.assert_array_equal(y_from_file, expected_y_sorted,
                                      err_msg="labelData.json 的内容与预期（排序后）不符。")
        print("OK: labelData.json 内容经过了正确的排序。")

        # 步骤 6: 验证 metadata.json 的内容
        with open(os.path.join(self.output_path, 'metadata.json'), 'r') as f:
            metadata_from_file = json.load(f)
            self.assertIn('sfreq', metadata_from_file)
            self.assertIn('classes', metadata_from_file)
            self.assertEqual(list(metadata_from_file['classes'].keys()), list(events_with_labels.keys()),
                             "元数据中的类别与定义的事件不符。")
        print("OK: metadata.json 文件内容有效。")


if __name__ == '__main__':
    # 允许直接从命令行运行此测试文件
    unittest.main(verbosity=2)
