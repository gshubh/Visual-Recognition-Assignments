import numpy as np
import pandas as pd
import cv2
import pickle
import cpickle as pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.cluster import KMeans


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# In[10]:


data = []
labels = []
for i in range(1,6):
    f = "cifar-10-python/ciphar-10-batches-py/data_batches_"+str(i)
    dict = unpickle(f)
    data.append(dict['data'])    
    labels.append(dict['labels'])


# In[2]:


# defining feature extractor that we want to use
extractor = cv2.xfeatures2d.SIFT_create()
def features(image, extractor):
    keypoints, descriptors = extractor.detectAndCompute(image, None)
    return keypoints, descriptors


# In[3]:


horses=[]
for i in range(1,81):
    s="Horses/horse"+str(i)+".jpg"
    img = cv2.imread(s,1)
    horses.append(img)


# In[4]:


bikes=[]
for i in range(10,81):
    s="Bikes/00"+str(i)+".jpg"
    img = cv2.imread(s,1)
    bikes.append(img)


# In[5]:


descriptor_list = []
for image in horses:
    keypoint, descriptor = features(image, extractor)
    for j in descriptor:
        descriptor_list.append(j)
for image in bikes:
    keypoint, descriptor = features(image, extractor)
    for j in descriptor:
        descriptor_list.append(j)


# In[7]:


kmeans = KMeans(n_clusters = 100)
kmeans.fit(descriptor)


# In[8]:


def build_histogram(descriptor_list, cluster_alg):
    histogram = np.zeros(len(cluster_alg.cluster_centers_))
    cluster_result =  cluster_alg.predict(descriptor_list)
    for i in cluster_result:
        histogram[i] += 1.0
    return histogram


# In[9]:


preprocessed_image = []

for image in horses:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoint, descriptor = features(image, extractor)
    if (descriptor is not None):
        histogram = build_histogram(descriptor, kmeans)
        preprocessed_image.append(histogram) 
for image in bikes:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoint, descriptor = features(image, extractor)
    if (descriptor is not None):
        histogram = build_histogram(descriptor, kmeans)
        preprocessed_image.append(histogram)


# In[10]:


data_frame = pd.DataFrame(preprocessed_image)
l = [0]*80 + [1]*71
data_frame['Label']=pd.Series(l)


# In[11]:


from sklearn.utils import shuffle
shuffled = shuffle(data_frame)


# In[12]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(shuffled, test_size = 0.3)


# In[13]:


x_train = train.drop('Label',1)
y_train = train.Label
x_test = test.drop('Label',1)
y_test = test.Label


# In[14]:


# Logistic Regression Model
lr_classifier = LogisticRegression(random_state=0)
lr_classifier.fit(x_train, y_train.values.ravel())
lr_classifier.score(x_test,y_test)


# In[15]:


# SVM model 
svclassifier = SVC(kernel='linear')  
svclassifier.fit(x_train, y_train.values.ravel())
svclassifier.score(x_test,y_test)


# In[19]:


# 
from sklearn.neighbors import KNeighborsClassifier  
knn_classifier = KNeighborsClassifier(n_neighbors=3)  
knn_classifier.fit(x_train, y_train.values.ravel())
knn_classifier.score(x_test,y_test)

