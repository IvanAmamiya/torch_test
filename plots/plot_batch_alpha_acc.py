import pandas as pd
import matplotlib.pyplot as plt
import re
import glob

# 搜索所有结果csv
csv_files = glob.glob('plots/mixup_alpha_results_*.csv')

batch_sizes = []
alpha_list = None
acc_dict = {}

for file in csv_files:
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # 提取batch_size
        for line in lines:
            if line.startswith('#batch_size='):
                batch_size = int(line.strip().split('=')[1])
                batch_sizes.append(batch_size)
                break
        # 提取alpha_list
        for line in lines:
            if line.startswith('#alpha_list=') or 'alpha_list=' in line:
                alpha_list = re.findall(r'\[(.*?)\]', line)
                if alpha_list:
                    alpha_list = [float(x) for x in alpha_list[0].split(',')]
                break
    # 用pandas读取数据部分，跳过注释和非数据行
    with open(file, 'r', encoding='utf-8') as f:
        data_start = 0
        for i, line in enumerate(f):
            if line.startswith('epoch,'):
                data_start = i
                break
    df = pd.read_csv(file, skiprows=data_start)
    # 只保留最后6行（即最后6个epoch，适用于30回实验）
    df_last = df.tail(6)
    # 计算每个alpha的acc均值
    accs = []
    for col in df.columns:
        if '_acc' in col:
            accs.append(df_last[col].mean())
    acc_dict[batch_size] = accs

# 绘图
plt.figure(figsize=(8,6))
for batch_size, accs in sorted(acc_dict.items()):
    plt.plot(alpha_list, accs, marker='o', label=f'batch_size={batch_size}')
plt.xlabel('alpha')
plt.ylabel('accuracy (30回均值)')
plt.title('不同batch_size下alpha与accuracy关系（30回均值）')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/batch_alpha_acc_curve.png')
plt.show()
