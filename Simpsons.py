

import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
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

def read_main(path):
    images,labels,name=read_image_labels(path,i=0)
    images=np.array(images,dtype=np.float32)/255
    labels=to_categorical(labels,num_classes=20)
    np.savetxt('name.txt',name,delimiter = ' ',fmt="%s")
    return images,labels
images,labels=read_main('autumn-classification/train/characters-20')
# X_train,X_test,y_train,y_test=train_test_split(images,labels,test_size=0.1)

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
            featurewise_center=False,  # 将输入数据的均值设置为 0，逐特征进行
            samplewise_center=False,  # 将每个样本的均值设置为 0
            featurewise_std_normalization=False,  # 将输入除以数据标准差，逐特征进行
            samplewise_std_normalization=False,  # 将每个输入除以其标准差
            zca_whitening=False,  # 应用 ZCA 白化
            rotation_range=10,  # 随机旋转的度数范围(degrees, 0 to 180)，旋转角度
            width_shift_range=0.1,  # 随机水平移动的范围，比例
            height_shift_range=0.1,  # 随机垂直移动的范围，比例
            horizontal_flip=True,  # 随机水平翻转，相当于镜像
            vertical_flip=False)  # 随机垂直翻转，相当于镜像
datagen.fit(X_train)


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=images.shape[1:]))#padding=same 输出与原始图像大小相同
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
opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)#nesterov=True使用动量
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics= ['accuracy'] )
# datagen=ImageDataGenerator(zoom_range=0.1,width_shift_range=0.05,height_shift_range=0.05,horizontal_flip=True)
datagen = ImageDataGenerator(
            zoom_range=0.1,
            featurewise_center=False,  # 将输入数据的均值设置为 0，逐特征进行
            samplewise_center=False,  # 将每个样本的均值设置为 0
            featurewise_std_normalization=False,  # 将输入除以数据标准差，逐特征进行
            samplewise_std_normalization=False,  # 将每个输入除以其标准差
            zca_whitening=False,  # 应用 ZCA 白化
            rotation_range=10,  # 随机旋转的度数范围(degrees, 0 to 180)，旋转角度
            width_shift_range=0.1,  # 随机水平移动的范围，比例
            height_shift_range=0.1,  # 随机垂直移动的范围，比例
            horizontal_flip=True,  # 随机水平翻转，相当于镜像
            vertical_flip=False)  # 随机垂直翻转，相当于镜像
datagen.fit(images)

file_name=str(epochs)+'_'+str(batch_size)
fig=model.fit_generator(datagen.flow(images,labels,batch_size=batch_size),
                        steps_per_epoch=epochs,
                        epochs=epochs,
                        validation_data=(images,labels),verbose=1)

model.save('h5/'+file_name+'.h5')
score=model.evaluate(images,labels,verbose=1)
print(score)

plt.figure()
print(fig.history.keys())
plt.plot(fig.history['loss'])
plt.plot(fig.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train','test'],loc='upper right')

plt.figure()
print(fig.history.keys())
plt.plot(fig.history['accuracy'])
plt.plot(fig.history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train','test'],loc='lower right')
plt.show()


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
