import subprocess
import time

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
    time.sleep(1000)  # 每30秒检查一次