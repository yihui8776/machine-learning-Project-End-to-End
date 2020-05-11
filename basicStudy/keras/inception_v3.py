# -*- coding:utf8 -*-

import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import decode_predictions

# 新建模型，此处实际上是导入预训练模型
model = InceptionV3()

# 预测函数
def predict(model, img):

    img = img.resize((299, 299))
    # 提取特征
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # 预测并输出概率最高的三个类别
    preds = model.predict(x)
    return decode_predictions(preds, top=3)[0]

# 绘制图像

def plot_preds(image, preds):

    plt.figure()
    plt.subplot(211)
    plt.imshow(image)

    plt.subplot(212)
    X1 = list(reversed(range(len(preds))))
    bar_preds = [pr[2] for pr in preds]
    labels = (pr[1] for pr in preds)
    plt.barh(X1, bar_preds, alpha=0.5)

    plt.yticks(X1, labels)
    plt.xlabel('Probability')
    plt.xlim(0,1.1)
    plt.tight_layout()
    plt.show()

# 运行
if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("-i")
    args = a.parse_args()

    img = Image.open(args.i)
    preds = predict(model, img)
    print(preds)
    plot_preds(img, preds)