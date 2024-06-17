#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import cv2
import os


# # Reading and Displaying an Image

# In[2]:


img=cv2.imread("E:\image.jpg.jpg")


# # Displaying Image Shape and Content

# In[3]:


img.shape


# In[4]:


img[0]


# In[5]:


img


# In[6]:


import matplotlib.pyplot as plt


# # Displaying the Image

# In[7]:


plt.imshow(img)


# # Installing Required Packages

# In[8]:


pip install opencv-python


# In[9]:


get_ipython().run_cell_magic('cmd', '', 'pip install cmake')


# In[10]:


pip install opencv-python


# In[11]:


pip install opencv-contrib-python


# In[12]:


import numpy as np


# # Creating a Window and Displaying the Image

# In[ ]:


while True:
    cv2.imshow('result',img)
    if cv2.waitKey(2)==27:
         break
cv2.destroyAllWindows()   


# # Initializing Haar Cascade Classifier for Face Detection

# In[ ]:


haar_data=cv2.CascadeClassifier('C:/Users/Dell/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')


# In[ ]:


haar_data.detectMultiScale(img)


# In[19]:


#cv2.rectangle(img,(x,y),(w,h),(b,g,r),border_thickness)


# # Displaying Detected Faces in Real-Time

# In[20]:


while True:
    faces=haar_data.detectMultiScale(img)
    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0))
    cv2.imshow('result',img)
    if cv2.waitKey(2)==27:
         break
cv2.destroyAllWindows()  


# # Real-Time Webcam Face Detection

# In[78]:


capture=cv2.VideoCapture(0)
while True:
    flag,img=capture.read()
    if flag:
        faces=haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0))
    cv2.imshow('result',img)
    if cv2.waitKey(2)==27:
         break
            
capture.release()
cv2.destroyAllWindows()  


# # Collecting Data for Face Mask Detection

# In[79]:


capture=cv2.VideoCapture(0)
data=[]
while True:
    flag,img=capture.read()
    if flag:
        faces=haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
            face=img[y:y+h,x:x+w, :]
            face=cv2.resize(face,(50,50))
            print(len(data))
            if len(data)<400:
                data.append(face)
    cv2.imshow('result',img)
    if cv2.waitKey(2)==27 or len(data)>=200:
         break
cv2.destroyAllWindows()  


# In[80]:


capture=cv2.VideoCapture(0)
data=[]
while True:
    flag,img=capture.read()
    if flag:
        faces=haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
            face=img[y:y+h,x:x+w, :]
            face=cv2.resize(face,(50,50))
            print(len(data))
            if len(data)<400:
                data.append(face)
    cv2.imshow('result',img)
    if cv2.waitKey(2)==27 or len(data)>=200:
         break
cv2.destroyAllWindows()  


# # Saving Data

# In[81]:


np.save('without_mask.npy',data)


# In[82]:


np.save('with1_mask.npy',data)


# In[83]:


import numpy as np


# In[84]:


import matplotlib.pyplot as plt


# In[85]:


plt.imshow(data[20])


# # Loading and Preprocessing Data for Training

# In[87]:


with_mask=np.load('with1_mask.npy')
without_mask=np.load('without_mask.npy')


# In[88]:


with_mask.shape


# In[89]:


without_mask.shape


# In[90]:


with_mask=with_mask.reshape(200,50*50*3)
without_mask=without_mask.reshape(200,50*50*3)


# In[91]:


without_mask.shape


# In[93]:


x=np.r_[with_mask,without_mask]


# # Labels and Training an SVM Classifier

# In[94]:


labels=np.zeros(x.shape[0])


# In[95]:


names={0:'Mask',1:'NoMask'}


# In[96]:


labels[200:]=1.0


# # Training and Testing an SVM Classifier

# In[97]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# In[99]:


from sklearn.model_selection import train_test_split


# In[100]:


x_train,x_test,y_train,y_test=train_test_split(x,labels,test_size=0.15)


# In[101]:


x_train.shape


# # Principal Component Analysis (PCA)

# In[102]:


from sklearn.decomposition import PCA


# In[103]:


pca=PCA(n_components=3)
x_train=pca.fit_transform(x_train)
x_train[0]


# In[104]:


svm=SVC()
svm.fit(x_train,y_train)


# In[105]:


x_test=pca.transform(x_test)
y_pred=svm.predict(x_test)


# In[106]:


accuracy_score(y_test,y_pred)


# # Real-Time Face Mask Detection

# In[107]:


haar_data=cv2.CascadeClassifier('C:/Users/Dell/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
capture=cv2.VideoCapture(0)
data=[]
font=cv2.FONT_HERSHEY_COMPLEX
while True:
    flag,img=capture.read()
    if flag:
        faces=haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
            face=img[y:y+h,x:x+w, :]
            face=cv2.resize(face,(50,50))
            face=face.reshape(1,-1)
            face= pca.transform(face)
            pred=svm.predict(face)
            n=names[int(pred)]
            cv2.putText(img,n,(x,y),font,1,(244,250,250),2)
            print(n)
    cv2.imshow('result',img)
    if cv2.waitKey(2)==27:
         break
cv2.destroyAllWindows()  


# In[ ]:




