import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

cred = credentials.Certificate('iot-project-kaiyang-firebase-adminsdk-r00dg-9e24793ca5.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# 添加数据
doc_ref = db.collection('users').document('user_id')
doc_ref.set({
    'name': 'John Doe',
    'email': 'johndoe@example.com'
})

# 获取数据
users_ref = db.collection('users')
docs = users_ref.stream()

for doc in docs:
    print(f'{doc.id} => {doc.to_dict()}')

