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

# 先push一次
git_push()

while True:
    time.sleep(30)  # 每30秒检查一次
    util = get_gpu_util()
    print(f"GPU util: {util}%")
    # 检查是否“暴降”
    if last_util > 50 and util < 10:
        print("GPU利用率暴降，自动git push")
        git_push()
        break
    last_util = util