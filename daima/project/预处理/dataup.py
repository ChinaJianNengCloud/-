#数据增强


#增加到10000个，一开始有3677个0以及1356个1
#要增加1323个0以及3644个1


#流程找出0和1的CR文件，如果是0的话，则水平翻转的创建一个新的文件夹，如果是1的话，则先水平翻转增加一倍，再90°旋转935个CR文件
#
import os
import numpy as np
import pandas as pd
from PIL import Image
data = pd.read_excel('duocengMUBA_rem.xlsx')

data1=data['影像号']
lable=data['良/恶']
path=r'36,312,5020new2'

file_name=os.listdir(path)
lable1=[]
for i in file_name:
    for  index ,j  in enumerate(data1):
        if i==j:
            lable1.append(lable[index])
data_end={'1':file_name,'2':lable1}
data_end = pd.DataFrame(data_end)
print(data_end['1'][1])
path_son=os.listdir(path)
for  index,i in enumerate(path_son):
    path_all=os.path.join(path,i)
    # 增加一倍的1，左右翻转
    if data_end['2'][index]==1:
        path_wenjian = path + os.sep + i + 'left_right1'
        if not os.path.exists(path_wenjian):
            os.makedirs(path_wenjian)
        for filename in os.listdir(path_all):
            if filename.endswith(('.jpg', '.png', '.jpeg', '.bmp', '.gif')):
                img_path = os.path.join(path_all, filename)
                img = Image.open(img_path)
                flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)

                flipped_img_path = os.path.join(path_wenjian, f'flipped_{filename}')
                flipped_img.save(flipped_img_path)
                print(f'Flipped {filename} and saved as {flipped_img_path}')
countshun=0

for index, i in enumerate(path_son):
    if countshun <= 1356:
        path_all = os.path.join(path, i)
    # 增加1356个顺时针翻转的1
        if data_end['2'][index]==1:
            countshun=countshun+1
            path_wenjian = path + os.sep + i + 'shun901'
            if not os.path.exists(path_wenjian):
                os.makedirs(path_wenjian)

            for filename in os.listdir(path_all):
                if filename.endswith(('.jpg', '.png', '.jpeg', '.bmp', '.gif')):
                    img_path = os.path.join(path_all, filename)
                    img = Image.open(img_path)
                    rotated_img = img.rotate(-90, expand=True)  # 顺时针旋转90度
                    rotated_img_path = os.path.join(path_wenjian, f'rotated_{filename}')
                    rotated_img.save(rotated_img_path)
                    print(f'Rotated {filename} and saved as {rotated_img_path}')
    else:
        break
countni=0
for index, i in enumerate(path_son):
    if countni <= 932:
        path_all = os.path.join(path, i)
            # 增加932个逆时针翻转的1
        if data_end['2'][index] == 1:
            countni = countni + 1
            path_wenjian = path + os.sep + i + 'ni901'
            if not os.path.exists(path_wenjian):
                os.makedirs(path_wenjian)

            for filename in os.listdir(path_all):
                if filename.endswith(('.jpg', '.png', '.jpeg', '.bmp', '.gif')):
                    img_path = os.path.join(path_all, filename)
                    img = Image.open(img_path)
                    rotated_img = img.rotate(90, expand=True)  # 逆时针旋转90度
                    rotated_img_path = os.path.join(path_wenjian, f'rotated_{filename}')
                    rotated_img.save(rotated_img_path)
                    print(f'Rotated {filename} and saved as {rotated_img_path}')
    else:
        break

    #增加1326个左右翻转的0
countlr=0
for index, i in enumerate(path_son):
    if countlr <= 1326:
        path_all = os.path.join(path, i)
        if data_end['2'][index]==0:
            countlr=countlr+1
            path_wenjian = path + os.sep + i + 'left_right0'
            if not os.path.exists(path_wenjian):
                os.makedirs(path_wenjian)

            for filename in os.listdir(path_all):
                if filename.endswith(('.jpg', '.png', '.jpeg', '.bmp', '.gif')):
                    img_path = os.path.join(path_all, filename)
                    img = Image.open(img_path)
                    flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)

                    flipped_img_path = os.path.join(path_wenjian, f'flipped_{filename}')
                    flipped_img.save(flipped_img_path)
                    print(f'Flipped {filename} and saved as {flipped_img_path}')
    else:
        break