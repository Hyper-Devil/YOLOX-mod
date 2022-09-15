import shutil
import os
'''
txt_path:txt的文件地址，里面存放的是所要复制的文件名，但这个文件名不包含文件地址
mat_path：所要复制的mat文件的文件地址
out_path:复制的目标地址

'''
txt_path='/whd/cone_coco2voc/val.txt'
mat_path=''
out_path='/whd/cone_coco2voc/val'
path_list=[]


with open(txt_path,'r') as f:
    for line in f.readlines():
        line = line.strip('\n')  #去掉列表中每一个元素的换行符
        path_list.append(line)

for i in path_list:
    # one_path = os.path.join(mat_path,i)
    one_path = i
    shutil.copy(one_path,out_path)
