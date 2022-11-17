import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2



### Load Model
model = load_model("dog_cat.h5")


# test_image = image.load_img('dataset/single_prediction/img1.jpeg', target_size = (64, 64))
img = cv2.imread("img3.jpeg")
test_image = cv2.resize(img, (64, 64))     ## resizing the data as per the train data

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)

if result[0][0] == 1:
  prediction = 'Dog'
else:
  prediction = 'Cat'

print(prediction)

### Put Text on predicted image
frame = cv2.putText(img, prediction, (0,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 5)

cv2.imshow("result", frame)
cv2.waitKey()
cv2.destroyAllWindows()