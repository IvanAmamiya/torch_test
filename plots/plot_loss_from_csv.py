import pandas as pd
import matplotlib.pyplot as plt
import time
import glob
import os

def plot_loss_from_csv(csv_file, save_dir='plots'):
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
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"mixup_alpha_loss_curve_{int(time.time())}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curve saved to {save_path}")

if __name__ == "__main__":
    # 自动查找最新的csv
    csv_files = sorted(glob.glob('plots/mixup_alpha_results_*.csv'), key=os.path.getmtime, reverse=True)
    if csv_files:
        plot_loss_from_csv(csv_files[0])
    else:
        print("No mixup_alpha_results_*.csv found.")
