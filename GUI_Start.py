import numpy as np
import pandas as pd
import json
import traceback
import os
import sys
import subprocess


# =======================================================================
# Part 1: 数据导出功能
# =======================================================================

def export_eeg_data():
    """
    加载、处理、排序并导出Bonn EEG数据集。
    如果所有步骤都成功，返回 True，否则返回 False。
    """
    try:
        print("=" * 60)
        print("🚀 Part 1: 开始执行数据导出任务...")
        print("=" * 60)
        print("--- 步骤 1: 正在加载数据... ---")
        print("这将需要一些时间，因为我们正在处理全部 300 个文件。")

        from metabci.brainda.datasets.bonn_eeg import BonnEEGDataset
        from metabci.brainda.paradigms.Bonn_paradigm import BonnEEGParadigm

        all_subjects = list(range(1, 101))
        events_with_labels = {'O': 0, 'F': 1, 'S': 2}
        dataset = BonnEEGDataset(path='Dataset/Bonn_EEG/')
        paradigm = BonnEEGParadigm(events=events_with_labels)

        x, y, meta = paradigm.get_data(
            dataset=dataset,
            subjects=all_subjects,
            return_concat=True,
            n_jobs=-1,
            verbose=False,
        )
        print("✅ 数据加载成功!")

        # --- 步骤 2: 数据验证与缩放 ---
        print("\n--- 步骤 2: 验证并缩放数据 ---")
        expected_trials = len(all_subjects) * len(events_with_labels)
        assert x.shape == (expected_trials, 1, 4096), "数据 x 的形状不符合预期！"
        assert y.shape == (expected_trials,), "标签 y 的形状不符合预期！"
        print("✅ 数据形状验证通过。")

        x = x / 1e6
        print("✅ 数据已通过除以 1e6 缩放回原始数值范围。")

        # --- 步骤 3: 按标签分组排序 ---
        print("\n--- 步骤 3: 按事件(标签)和被试排序数据 ---")
        meta.sort_values(by=['event', 'subject'], inplace=True)
        sorted_indices = meta.index
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]
        print("✅ 数据已根据事件类型和被试ID重新排序。")

        # --- 步骤 4: 导出为 JSON 文件到指定路径 ---
        print("\n--- 步骤 4: 导出为 JSON 文件到指定路径 ---")
        output_dir = os.path.join('metabci', 'brainda', 'data')
        os.makedirs(output_dir, exist_ok=True)
        print(f"✅ 确保输出目录存在: '{output_dir}'")

        sig_data_filename = os.path.join(output_dir, 'sigData.json')
        label_data_filename = os.path.join(output_dir, 'labelData.json')

        print(f"⏳ 正在保存信号数据到 '{sig_data_filename}'...")
        with open(sig_data_filename, 'w') as f:
            json.dump(x_sorted.tolist(), f)
        print(f"✅ '{sig_data_filename}' 保存成功!")

        print(f"⏳ 正在保存标签数据到 '{label_data_filename}'...")
        with open(label_data_filename, 'w') as f:
            json.dump(y_sorted.tolist(), f)
        print(f"✅ '{label_data_filename}' 保存成功!")

        print("\n🎉 数据导出任务全部完成！")
        return True

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"🔥 错误：数据导出过程中发生严重错误，无法继续。")
        print("=" * 60)
        traceback.print_exc()
        return False


# =======================================================================
# Part 2: GUI 启动功能
# =======================================================================

def run_gui_application():
    """
    一个健壮的启动器，用于从主项目目录运行子工程中的GUI应用。
    """
    try:
        print("\n" + "=" * 60)
        print("🚀 Part 2: 准备启动GUI应用程序...")
        print("=" * 60)

        # 获取此启动器脚本所在的目录 (即项目根目录)
        launcher_dir = os.path.dirname(os.path.abspath(__file__))

        # 构建到GUI子工程目录和其主脚本的路径
        gui_project_dir = os.path.join(launcher_dir, 'metabci', 'brainda', 'GUI')
        app_script_path = os.path.join(gui_project_dir, 'main.py')

        # --- 提供友好的错误检查 ---
        if not os.path.isdir(gui_project_dir):
            print(f"❌ 错误：未找到GUI子工程目录。")
            print(f"   期望路径: {gui_project_dir}")
            return
        if not os.path.isfile(app_script_path):
            print(f"❌ 错误：在GUI目录中未找到主程序 'main.py'。")
            print(f"   期望路径: {app_script_path}")
            return

        # --- 执行启动 ---
        print(f"✅ 找到GUI应用, 准备启动...")
        print(f"   > 目标目录: {gui_project_dir}")
        print(f"   > 执行脚本: {app_script_path}")

        # 使用 subprocess.run 来执行脚本
        # 'cwd' 参数是这里的核心，它将子进程的工作目录设置为GUI的目录
        # 'sys.executable' 确保我们使用与启动器相同的Python解释器
        subprocess.run(
            [sys.executable, app_script_path],
            cwd=gui_project_dir,
            check=True  # 如果子进程返回错误代码，则会抛出异常
        )

    except subprocess.CalledProcessError as e:
        print("\n" + "-" * 60)
        print(f"❗️ GUI应用程序异常退出，返回错误码: {e.returncode}")
    except Exception as e:
        print("\n" + "-" * 60)
        print(f"🔥 GUI启动器发生未知错误: {e}")
        traceback.print_exc()


# =======================================================================
# Part 3: 主执行流程
# =======================================================================

if __name__ == '__main__':
    # 首先，执行数据导出功能
    is_data_ready = export_eeg_data()

    # 然后，检查数据是否准备就绪。如果是，则启动GUI
    if is_data_ready:
        run_gui_application()
    else:
        print("\n" + "=" * 60)
        print("❌ 任务终止：由于数据导出失败，GUI应用程序将不会启动。")
        print("=" * 60)
        # 暂停一下，让用户能看到错误信息
        input("按 Enter 键退出...")


