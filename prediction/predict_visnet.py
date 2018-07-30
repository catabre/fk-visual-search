import os
import sys
import numpy as np
   
   #Set up caffe root directory and add to path
  # caffe_root = '$APPS/caffe/'
  # sys.path.insert(0, caffe_root + 'python')
  # sys.path.append('opencv-2.4.13/release/lib/')
   
import cv2
import caffe
import traceback
from sklearn.neighbors import KNeighborsClassifier

  # Parameters
#scale = 3
  
  #Create Caffe model using pretrained model
caffe_root = '/home/300002291/fk-visual-search/'

try:
    net = caffe.Net(caffe_root + 'models/visnet/train2.prototxt',
                    caffe_root + 'snapshots/visnet-s2s/_iter_0.caffemodel.h5', caffe.TEST)
except:
   traceback.print_exc() 
  
print ("Initializarion of the Caffe Net is Done")
  #Input directories
#input_dir = caffe_root + 'data/images/'
  
  #Input ground truth image
im_raw = cv2.imread(caffe_root + 'prediction/10569.jpg')
  
  #Change format to YCR_CB
ycrcb = cv2.cvtColor(im_raw, cv2.COLOR_RGB2YCR_CB)
im_raw = ycrcb[:,:,0]
#im_raw = im_raw.reshape((im_raw.shape[0], im_raw.shape[1], 1))
  
  #Blur image and resize to create input for network
#im_blur = cv2.blur(im_raw, (4,4))
im_small = cv2.resize(im_raw, (224, 224))

#im_raw = im_raw.reshape((1, 1, im_raw.shape[0], im_raw.shape[1]))
#im_blur = im_blur.reshape((1, 1, im_blur.shape[0], im_blur.shape[1]))
#im_small = im_small.reshape((1, 1, im_small.shape[0], im_small.shape[1]))
  
#im_comp = im_blur
#im_input = im_small
  
  #Set mode to run on CPU
caffe.set_mode_cpu()
im_input = [im_small]
  #Copy input image data to net structure
#c1,c2,h,w = im_input.shape
net.blobs['data_p'].data[...] = im_input
net.blobs['data_q'].data[...] = im_input
net.blobs['data_n'].data[...] = im_input
  
  #Run forward pass
print("Running the forward pass of the net ")
out = net.forward()
print("Eecution Completed")
  
  #Extract output image from net, change format to int8 and reshape
print("Output::" + str(out['linear_embedding_p_norm']))
print("Size: " + str(out['linear_embedding_p_norm'].shape))

knn = KNeighborsClassifier(n_neighbors=5)
X_train = out['linear_embedding_p_norm']
n, x, y, z = out['linear_embedding_p_norm'].shape
X_train = X_train.reshape((x))
y_train = [0]
knn.fit(X_train, y_train)

#mat = out['linear_embedding_p_norm']
#mat = (mat[0,:,:]).astype('uint8')
  
#im_raw = im_raw.reshape((im_raw.shape[2], im_raw.shape[3]))
#im_blur = im_blur.reshape((im_blur.shape[2], im_blur.shape[3]))
#im_comp = im_blur.reshape((im_comp.shape[2], im_comp.shape[3]))
  
  #Display original (ground truth), blurred and restored images
#cv2.imshow("image",im_raw)
#cv2.imshow("image2",im_comp)
#cv2.imshow("image3",mat)
#cv2.waitKey()
  
#cv2.destroyAllWindows()

