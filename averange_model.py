import tensorflow as tf
from google.cloud import storage
import os
import firebase_admin
from firebase_admin import credentials, firestore

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical

# 初始化 Firebase
cred = credentials.Certificate('iot-project-kaiyang-firebase-adminsdk-r00dg-9e24793ca5.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# 定义从 Google Cloud Storage 下载模型的函数
def download_model_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """从 Google Cloud Storage 下载模型文件。"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)
    print(f"Model {source_blob_name} downloaded to {destination_file_name}.")

def upload_model_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """上传模型文件到 Google Cloud Storage。"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    return f"gs://{bucket_name}/{destination_blob_name}"
# 下载模型
bucket_name = 'iot-project-bucket1'
model_filenames = ['mnist_model_1.h5', 'mnist_model_2.h5', 'mnist_model_3.h5']
local_model_paths = []

for filename in model_filenames:
    local_path = f'local_{filename}'
    download_model_from_gcs(bucket_name, filename, local_path)
    local_model_paths.append(local_path)

# 加载模型并计算平均权重
models = [tf.keras.models.load_model(path) for path in local_model_paths]
weights = [model.get_weights() for model in models]

# 计算权重平均值
average_weights = [sum(w) / len(w) for w in zip(*weights)]

# 创建新模型并应用平均权重
avg_model = tf.keras.models.clone_model(models[0])
avg_model.set_weights(average_weights)

# 保存平均后的模型
avg_model_path = 'avg_model.h5'
avg_model.save(avg_model_path)

# 上传平均后的模型
gcs_avg_model_path = upload_model_to_gcs(bucket_name, avg_model_path, 'avg_mnist_model.h5')

# 将平均模型链接保存到 Firestore
doc_ref = db.collection('average_models').document('avg_mnist_model')
doc_ref.set({'model_url': gcs_avg_model_path})

print("Average model uploaded to GCS and link saved to Firestore.")



# ...（前面的代码，包括加载和平均模型的部分）

# 加载 MNIST 测试数据集（如果你之前已经加载过，就不需要重复这一步）
(test_images, test_labels) = mnist.load_data()[1]
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
test_labels = to_categorical(test_labels)


avg_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# 使用平均后的模型对测试数据进行评估
loss, accuracy = avg_model.evaluate(test_images, test_labels)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")


# 清理本地文件
for path in local_model_paths:
    os.remove(path)
os.remove(avg_model_path)



