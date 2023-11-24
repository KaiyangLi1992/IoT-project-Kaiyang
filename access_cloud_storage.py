import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud import storage
import os

# 初始化 Firebase
cred = credentials.Certificate('iot-project-kaiyang-firebase-adminsdk-r00dg-9e24793ca5.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# 加载 MNIST 数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 定义上传模型到 Google Cloud Storage 的函数
def upload_model_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """上传模型文件到 Google Cloud Storage。"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    return f"gs://{bucket_name}/{destination_blob_name}"

# 训练并保存多个模型
for i in range(1, 4):  # 生成三个模型
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=50)

    # 保存模型
    model_path = f'my_model_{i}.h5'
    model.save(model_path)

    # 上传模型
    bucket_name = 'iot-project-bucket1'  # Cloud Storage 桶名称
    gcs_model_path = upload_model_to_gcs(bucket_name, model_path, f'mnist_model_{i}.h5')

    # 将模型链接保存到 Firestore
    doc_ref = db.collection('models').document(f'mnist_model_{i}')
    doc_ref.set({'model_url': gcs_model_path})

    print(f"Model {i} uploaded to GCS and link saved to Firestore.")

    # 清理本地文件
    os.remove(model_path)
