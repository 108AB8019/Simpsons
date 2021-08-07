# NTUT-2020-Autumn-Classification
## Step1 使用環境
```python=
import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
import os,sys
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt 
import pandas as pd
```
## Step2 讀取資料及處理
```python=
#讀取資料及處理
def read_image_labels(path,i):
    for file in os.listdir(path):
        abs_path=os.path.abspath(os.path.join(path,file))
        if os.path.isdir(abs_path):
            i+=1
            temp=os.path.split(abs_path)[-1]
            name.append(temp)
            read_image_labels(abs_path,i)
            amount=int(len(os.listdir(path)))
            sys.stdout.write('\r'+'>'+ ' '*(amount-i)+'[%s%%]'%(i*100/amount)+temp)
        else :
            if file.endswith('.jpg'):
                image=cv2.resize(cv2.imread(abs_path),(64,64))
                images.append(image)
                labels.append(i-1)
    return images,labels,name
```
```python=
def read_main(path):
    images,labels,name=read_image_labels(path,i=0)
    images=np.array(images,dtype=np.float32)/255
    labels=to_categorical(labels,num_classes=20)
    np.savetxt('name.txt',name,delimiter = ' ',fmt="%s")
    return images,labels
images,labels=read_main('autumn-classification/train/characters-20')
```
## Step3 模型訓練
### Step 3-1 模型一
```python=
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=images.shape[1:]))#padding=same 輸出與原始大小相同
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), padding='same')) 
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))


model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(20, activation='softmax'))
opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)#nesterov=True使用動量
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics= ['accuracy'] )
# datagen=ImageDataGenerator(zoom_range=0.1,width_shift_range=0.05,height_shift_range=0.05,horizontal_flip=True)
datagen = ImageDataGenerator(
            zoom_range=0.1,
            featurewise_center=False,  # 將輸入數據均值設為0，逐筆特徵值進行
            samplewise_center=False,  # 將每個樣本的值設置为 0
            featurewise_std_normalization=False,  # 將輸入除以數據標準差，逐筆特徵值進行
            samplewise_std_normalization=False,  # 將每個輸入除以標準差
            zca_whitening=False,  # 應用ZCA白話
            rotation_range=10,  # 隨機旋轉的度數範圍(0 to 180)，旋轉角度
            width_shift_range=0.1,  # 隨機水平移動的範圍，比例
            height_shift_range=0.1,  # 隨機垂直移動的範圍，比例
            horizontal_flip=True,  # 隨機水平翻轉，相當於鏡像
            vertical_flip=False)  # 隨機垂直翻轉，相當於鏡像
datagen.fit(images)
```

##### 層數資訊
```
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_6 (Conv2D)            (None, 64, 64, 32)        896       
_________________________________________________________________
activation_7 (Activation)    (None, 64, 64, 32)        0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 62, 62, 32)        9248      
_________________________________________________________________
activation_8 (Activation)    (None, 62, 62, 32)        0         
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 31, 31, 32)        0         
_________________________________________________________________
dropout_4 (Dropout)          (None, 31, 31, 32)        0         
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 31, 31, 64)        18496     
_________________________________________________________________
activation_9 (Activation)    (None, 31, 31, 64)        0         
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 29, 29, 64)        36928     
_________________________________________________________________
activation_10 (Activation)   (None, 29, 29, 64)        0         
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 14, 14, 64)        0         
_________________________________________________________________
dropout_5 (Dropout)          (None, 14, 14, 64)        0         
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 14, 14, 256)       147712    
_________________________________________________________________
activation_11 (Activation)   (None, 14, 14, 256)       0         
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 12, 12, 256)       590080    
_________________________________________________________________
activation_12 (Activation)   (None, 12, 12, 256)       0         
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 6, 6, 256)         0         
_________________________________________________________________
dropout_6 (Dropout)          (None, 6, 6, 256)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 9216)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 1024)              9438208   
_________________________________________________________________
activation_13 (Activation)   (None, 1024)              0         
_________________________________________________________________
dropout_7 (Dropout)          (None, 1024)              0         
_________________________________________________________________
dense_3 (Dense)              (None, 20)                20500     
=================================================================
Total params: 10,262,068
Trainable params: 10,262,068
Non-trainable params: 0
_________________________________________________________________
```

#### 模型收斂狀況
```
Epoch 45/50
50/50 [==============================] - 18s 368ms/step - loss: 0.1961 - accuracy: 0.9384 - val_loss: 0.0685 - val_accuracy: 0.9822
Epoch 46/50
50/50 [==============================] - 18s 369ms/step - loss: 0.1844 - accuracy: 0.9434 - val_loss: 0.0592 - val_accuracy: 0.9846
Epoch 47/50
50/50 [==============================] - 18s 365ms/step - loss: 0.1872 - accuracy: 0.9424 - val_loss: 0.0669 - val_accuracy: 0.9812
Epoch 48/50
50/50 [==============================] - 18s 365ms/step - loss: 0.1665 - accuracy: 0.9496 - val_loss: 0.0561 - val_accuracy: 0.9853
Epoch 49/50
50/50 [==============================] - 18s 366ms/step - loss: 0.1742 - accuracy: 0.9451 - val_loss: 0.0574 - val_accuracy: 0.9844
Epoch 50/50
50/50 [==============================] - 18s 365ms/step - loss: 0.1822 - accuracy: 0.9457 - val_loss: 0.0717 - val_accuracy: 0.9788
```
![](https://i.imgur.com/oAGSBRk.png)
![](https://i.imgur.com/2Z0vGKP.png)
#### 儲存模型
```python=
model.save('h5/'+file_name+'.h5')
```
### Step 3-2 模型二
```python=
seed = 7
np.random.seed(seed)
 
model = Sequential()
model.add(Conv2D(96,(11,11),strides=(4,4),input_shape=X_train.shape[1:],padding='valid',activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20,activation='softmax'))
opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)#nesterov=True使用动量
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
model.summary()

# datagen=ImageDataGenerator(zoom_range=0.1,width_shift_range=0.05,height_shift_range=0.05,horizontal_flip=True)
datagen = ImageDataGenerator(
            zoom_range=0.1,
            featurewise_center=False,  # 將輸入數據均值設為0，逐筆特徵值進行
            samplewise_center=False,  # 將每個樣本的值設置为 0
            featurewise_std_normalization=False,  # 將輸入除以數據標準差，逐筆特徵值進行
            samplewise_std_normalization=False,  # 將每個輸入除以標準差
            zca_whitening=False,  # 應用ZCA白話
            rotation_range=10,  # 隨機旋轉的度數範圍(0 to 180)，旋轉角度
            width_shift_range=0.1,  # 隨機水平移動的範圍，比例
            height_shift_range=0.1,  # 隨機垂直移動的範圍，比例
            horizontal_flip=True,  # 隨機水平翻轉，相當於鏡像
            vertical_flip=False)  # 隨機垂直翻轉，相當於鏡像
datagen.fit(X_train)
```
##### 層數資訊
```
Model: "sequential_8"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_48 (Conv2D)           (None, 14, 14, 96)        34944     
_________________________________________________________________
max_pooling2d_25 (MaxPooling (None, 7, 7, 96)          0         
_________________________________________________________________
conv2d_49 (Conv2D)           (None, 7, 7, 256)         614656    
_________________________________________________________________
max_pooling2d_26 (MaxPooling (None, 3, 3, 256)         0         
_________________________________________________________________
conv2d_50 (Conv2D)           (None, 3, 3, 384)         885120    
_________________________________________________________________
conv2d_51 (Conv2D)           (None, 3, 3, 384)         1327488   
_________________________________________________________________
conv2d_52 (Conv2D)           (None, 3, 3, 256)         884992    
_________________________________________________________________
max_pooling2d_27 (MaxPooling (None, 1, 1, 256)         0         
_________________________________________________________________
flatten_8 (Flatten)          (None, 256)               0         
_________________________________________________________________
dense_18 (Dense)             (None, 4096)              1052672   
_________________________________________________________________
dropout_29 (Dropout)         (None, 4096)              0         
_________________________________________________________________
dense_19 (Dense)             (None, 4096)              16781312  
_________________________________________________________________
dropout_30 (Dropout)         (None, 4096)              0         
_________________________________________________________________
dense_20 (Dense)             (None, 20)                81940     
=================================================================
Total params: 21,663,124
Trainable params: 21,663,124
Non-trainable params: 0
_________________________________________________________________
```


## Step4 模型驗證
```python=
def read_images(path):
    images=[]
    for i in range(990):
        image=cv2.resize(cv2.imread(path+str(i+1)+'.jpg'),(64,64))
        images.append(image)
    images=np.array(images,dtype=np.float32)/255
    return images

def transform(listdir,label,lenSIZE):
    label_str=[]
    for i in range(lenSIZE):
        temp=listdir[label[i]]
        label_str.append(temp)
    return label_str

images=read_images('autumn-classification/test/test/')
# model=tf.keras.models.load_model('h5/50_128.h5')

predit=model.predict_classes(images,verbose=1)
print(predit)
label_str=transform(np.loadtxt('name.txt',dtype='str'),predit,images.shape[0])

df=pd.DataFrame({"character":label_str})
df.index=np.arange(1,len(df)+1)
df.index.names=['id']
df.to_csv('a.csv')
```

##### 最後分數輸出 使用模型一做為最後輸出
![](https://i.imgur.com/O50gWzj.png)

