import pandas as pd
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
import os
import numpy as np
import config as cfg

from keras.layers import Input, Dense



class Extractor():#特征提取器
    def __init__(self, weights=None):
        self.weights = weights  # so we can check elsewhere which model

        if weights is None:
            base_model = InceptionV3(weights='imagenet',include_top=True)#加载InceptionV3模型自带的预处理模型
            # We'll extract features at the final pool layer.
            self.model = Model(inputs=base_model.input,
                                outputs=base_model.get_layer('avg_pool').output)
        else:
            # Load the model first.
            self.model = load_model(weights)
            self.model.layers.pop()
            self.model.layers.pop()  # 提取特征不需要最后两层
            self.model.outputs = [self.model.layers[-1].output]#更新模型的属性
            self.model.output_layers = [self.model.layers[-1]]
            self.model.layers[-1].outbound_nodes = []

    def extract(self, file_path):

        all_features = []
        pic_path=os.listdir(file_path)
        for image_path in pic_path:
            image_tmp=str(file_path)+os.sep+str(image_path)
            img = image.load_img(image_tmp, target_size=(299,299))#加载图片路径
            x = image.img_to_array(img)#数组形式,代表像素值
            x = np.expand_dims(x, axis=0)#在数组的第0维加一个维度
            x = preprocess_input(x)
            # Get the prediction.
            features = self.model.predict(x)#特征提取
            all_features.append(features)
        all_features = np.array(all_features)#转numpy数组形式
        features_means = np.mean(all_features, axis=1)
        features_means=features_means.T
        # input_layer = Input(shape=(all_features,))
        # encoded = Dense(encoding_dim, activation='relu')(input_layer)

        # decoded = Dense(original_dim, activation='sigmoid')(encoded)
        return features_means#返回特征值

    def get_file(self):#获取图片路径
        file_path_tmp = cfg.Pic_dir
        file_name=os.listdir(file_path_tmp)
        if len(file_name) == 0:
            raise Exception("There is no file")
        else:
            for file in file_name:
                file_path = str(file_path_tmp)+os.sep+str(file)
                file_index=file
                yield file_index,file_path

    def process_img(self):
        p = self.get_file()
        while(1):
            try:
                file_index,file_path = next(p)
            except StopIteration:
                print("The extract-feature job finish")
                break
            features = self.extract(file_path)
            if not os.path.exists(cfg.Npy_dir):
                os.makedirs(cfg.Npy_dir)
            np.save(os.path.join(cfg.Npy_dir,file_index+'.npy'), features)

A=Extractor()
A.process_img()


