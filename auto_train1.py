import pynvml
import os
import time
pynvml.nvmlInit()
# 这里的0是GPU id
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
ratio = 1024**2
while 1:
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total = meminfo.total/ratio  # 以兆M为单位就需要除以1024**2
    used = meminfo.used/ratio
    free = meminfo.free/ratio
    print("total: ", total)
    print("used: ", used)
    print("free: ", free)
    if used < total/12:
        print("start")
        os.system('python -m yolox.tools.train -f exps/example/custom/bit.py -d 1 -b 32 --fp16 --cache -o -expn 113 --logger wandb wandb-project YOLOX wandb-name 113-bit+3ECA+Focalloss+HorBlock')
        print("finish")
        break
    time.sleep(300)
