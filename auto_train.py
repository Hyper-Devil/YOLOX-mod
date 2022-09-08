#coding: utf-8
from re import T
import pynvml
import os
import time

gpu_memory = 9 #预期GPU剩余显存，如果达到这个数字就运行train脚本
trainpythonfile= '1.py'
pynvml.nvmlInit()

gpu_list = [
    'os.environ["CUDA_VISIBLE_DEVICES"] = "0"',
    # 'os.environ["CUDA_VISIBLE_DEVICES"] = "1"',
    # 'os.environ["CUDA_VISIBLE_DEVICES"] = "2"',
    # 'os.environ["CUDA_VISIBLE_DEVICES"] = "3"',
    # 'os.environ["CUDA_VISIBLE_DEVICES"] = "4"',
    # 'os.environ["CUDA_VISIBLE_DEVICES"] = "5"',
    # 'os.environ["CUDA_VISIBLE_DEVICES"] = "6"',
    # 'os.environ["CUDA_VISIBLE_DEVICES"] = "7"'
]

# 读写python文件在指定位置插入显卡编号
def writepythonfile(pythonfile,cline,string):
    lines = []
    with open(pythonfile,"r") as y:
        for line in y:
            lines.append(line)
        y.close()
    lines.insert(cline, string+"\n")
    s = ''.join(lines)
    with open(pythonfile,"w") as z:
        z.write(s)
        z.close()
    del lines[:]

while True:
    time.sleep(300)
    for i in range(len(gpu_list)):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print('第'+str(i)+'块GPU剩余显存'+str(meminfo.free/(1024**3))+'GB') #第二块显卡剩余显存大小
        if meminfo.free/(1024**2)>=gpu_memory*1024:
            writepythonfile(trainpythonfile,0,'import os')
            writepythonfile(trainpythonfile,1,gpu_list[i])
            os.system('python 1.py')
            break
        else:
            print("不符合剩余"+str(gpu_memory)+"GB显存需求")