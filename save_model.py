import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import numpy as np
import os

# 初始化 Firebase
cred = credentials.Certificate('iot-project-kaiyang-firebase-adminsdk-r00dg-9e24793ca5.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# 加载 MNIST 数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 数据预处理
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 定义 CNN 模型
def create_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 训练并保存模型
for i in range(3):
    model = create_model()
    model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=5)

    # 保存模型到本地文件
    model_path = f'model_{i}.h5'
    model.save(model_path)

    # 将模型转换为二进制格式
    with open(model_path, "rb") as model_file:
        model_binary = model_file.read()

    # 上传到 Firebase
    doc_ref = db.collection('models').document(f'model_{i}')
    doc_ref.set({'model': model_binary})

    # 删除本地文件
    os.remove(model_path)

print("Models have been trained and uploaded to Firebase.")
