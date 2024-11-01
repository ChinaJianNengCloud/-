import os
Class_num = 2#分类个个数
Class = ['0','1']
Batch_size = 16
Input_seq_len = 16
Input_dim =36*2048
Pic_dir = r'E:\ww\duocengMUBA_rem\duocengMUBA_rem_256_312\duocengMUBA_rem'#读取图片的路径
Npy_dir = r'E:\yinda\ww\timesformer_1\datanpy'#存图片特征值的路径
Label_txt = r'./newinfo/label.txt'
Train_list_txt = r'./newinfo/train.txt'
Val_list_txt = r'./newinfo/val.txt'