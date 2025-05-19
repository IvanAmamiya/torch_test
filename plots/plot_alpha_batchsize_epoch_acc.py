import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# 搜索所有结果csv
csv_files = glob.glob('plots/mixup_alpha_results_*.csv')

# 记录所有(batch_size, alpha, epoch, acc, loss)
records = []

for file in csv_files:
    batch_size = None
    alpha_list = None
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('#batch_size='):
                batch_size = int(line.strip().split('=')[1])
            if line.startswith('#alpha_list=') or 'alpha_list=' in line:
                import re
                alpha_list = re.findall(r'\[(.*?)\]', line)
                if alpha_list:
                    alpha_list = [float(x) for x in alpha_list[0].split(',')]
    # 跳过注释行，找到数据起始行
    with open(file, 'r', encoding='utf-8') as f:
        data_start = 0
        for i, line in enumerate(f):
            if line.startswith('epoch,'):
                data_start = i
                break
    df = pd.read_csv(file, skiprows=data_start)
    # 遍历所有epoch、alpha
    for idx, row in df.iterrows():
        epoch = row['epoch']
        for i, alpha in enumerate(alpha_list):
            acc = row[f'alpha={alpha}_acc'] if f'alpha={alpha}_acc' in row else None
            loss = row[f'alpha={alpha}_loss'] if f'alpha={alpha}_loss' in row else None
            records.append({'batch_size': batch_size, 'alpha': alpha, 'epoch': epoch, 'acc': acc, 'loss': loss})

# 转为DataFrame
all_df = pd.DataFrame(records)

# 画三维曲线：x=alpha, y=epoch, z=acc，分batch_size画多条
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')
for batch_size in sorted(all_df['batch_size'].unique()):
    df_bs = all_df[all_df['batch_size']==batch_size]
    for alpha in sorted(df_bs['alpha'].unique()):
        df_alpha = df_bs[df_bs['alpha']==alpha]
        ax.plot([alpha]*len(df_alpha), df_alpha['epoch'], df_alpha['acc'], label=f'batch={batch_size}, alpha={alpha}')
ax.set_xlabel('Alpha')
ax.set_ylabel('Epoch')
ax.set_zlabel('Accuracy')
ax.set_title('Alpha-BatchSize-Epoch-Accuracy 曲线')
plt.legend()
plt.tight_layout()
plt.savefig('plots/alpha_batchsize_epoch_acc_3d.png')
plt.show()
