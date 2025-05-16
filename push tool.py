import subprocess
import time
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

def get_gpu_util():
    try:
        output = subprocess.check_output(
            "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits", shell=True
        )
        return int(output.decode().strip().split('\n')[0])
    except Exception:
        return 0

def git_push():
    subprocess.run("git add .", shell=True)
    subprocess.run("git commit -m '自动上传'", shell=True)
    subprocess.run("git push", shell=True)

last_util = get_gpu_util()

def gpu_has_process():
    output = subprocess.check_output("nvidia-smi", shell=True).decode()
    return "No running processes found" not in output

if gpu_has_process():
    print("GPU有任务在运行")
else:
    print("GPU空闲")

# 先push一次
git_push()

while True:
    util = get_gpu_util()
    print(f"GPU util: {util}%")
    # 检查是否“暴降”
    git_push()
    # 自动查找最新的csv并保存loss曲线
    csv_files = sorted(glob.glob('plots/mixup_alpha_results_*.csv'), key=os.path.getmtime, reverse=True)
    if csv_files:
        csv_file = csv_files[0]
        # 跳过注释行，找到数据起始行
        with open(csv_file, 'r', encoding='utf-8') as f:
            data_start = 0
            for i, line in enumerate(f):
                if line.startswith('epoch,'):
                    data_start = i
                    break
        df = pd.read_csv(csv_file, skiprows=data_start)
        alpha_cols = [col for col in df.columns if '_loss' in col]
        plt.figure(figsize=(10, 6))
        for col in alpha_cols:
            plt.plot(df['epoch'], df[col], marker='o', label=col.replace('_loss',''))
        plt.title('Mixup Alpha vs Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join('plots', f"mixup_alpha_loss_curve_{int(time.time())}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Loss curve saved to {save_path}")
    time.sleep(1000)  # 每30秒检查一次