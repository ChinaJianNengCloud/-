import pandas as pd
import config as cfg
import os
import numpy as np
# 从Excel文件中读取表格数据
data = pd.read_excel('duocengMUBA_rem.xlsx')

data1=data['影像号']
lable=data['良/恶']
path=r'36,312,5031new'
file_name=os.listdir(path)
# print(file_name)
lable1=[]
for i in file_name:

    for  index ,j  in enumerate(data1):
        if str(i)[0:9]==j:
            lable1.append(lable[index])
# print(lable1)
print(len(lable1),len(file_name))
if os.path.exists(cfg.Train_list_txt):
    print("文件夹已经存在")
else:
    os.mkdir(r"result\newinfo")

data_end={'1':file_name,'2':lable1}
#
# print(data_end)
data_end = pd.DataFrame(data_end)
# # data_end.to_csv('123.csv', sep='\t', index=False,header=False)
# data_end = pd.DataFrame(data_end)
# print(data_end)
train_file_name=file_name[0:9000]
val_file_name=file_name[9000:10000]
train=pd.DataFrame(train_file_name)
val=pd.DataFrame(val_file_name)
# #
data_end.to_csv(cfg.Label_txt, sep='\t', index=False,header=False)
train.to_csv(cfg.Train_list_txt, sep='\t', index=False,header=False)
val.to_csv(cfg.Val_list_txt, sep='\t', index=False,header=False)