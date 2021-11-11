import os

# **   ANTI SPOOFING       ** #
# Anti spoofing directories
train_dir = "antispoofing_dataset/train"
test_dir = "antispoofing_dataset/test"

# Dataset Exploration
categories = ["real", "spoof"]
print("---------------------Exploring Training Datasets--------------------")
for category in categories:
    path = os.path.join(train_dir, category)
    if category == 'real':
        r1 = len(os.listdir(path))
    else:
        s1 = len(os.listdir(path))
    print("There are {} images in {} directory".format(len(os.listdir(path)), category))
print("There are {} total images in training directory".format(r1 + s1))

print("-----------------------Exploring Testing Datasets-------------------------")
for category in categories:
    path = os.path.join(test_dir, category)
    if category == 'real':
        r2 = len(os.listdir(path))
    else:
        s2 = len(os.listdir(path))
    print("There are {} images in {} directory".format(len(os.listdir(path)), category))
print("There are {} total images in testing directory".format(r2 + s2))

#################################### Model Preparation ######################################

# Keras Framework
from keras.layers import Dense,Dropout,Input,Flatten
from keras.models import Model
from keras.optimizers import adam_v2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import model_from_json
import json


# Load datasets and Perform image augmentations
train_datagen = ImageDataGenerator(brightness_range=(0.8,1.2),rotation_range=30,width_shift_range=0.2,height_shift_range=0.2,fill_mode='nearest',shear_range=0.2,zoom_range=0.3,rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,target_size=(160,160),color_mode='rgb',
                                                    class_mode='binary',batch_size=25,shuffle=True)
valid_generator = valid_datagen.flow_from_directory(test_dir,target_size=(160,160),color_mode='rgb',
                                                    class_mode='binary',batch_size=25)
# Model Selection and Transfer Learning
mobilenet = MobileNetV2(weights="imagenet",include_top=False,input_tensor=Input(shape=(160,160,3)))
mobilenet.trainable = False

output = Flatten()(mobilenet.output)
output = Dropout(0.3)(output)
output = Dense(units = 8,activation='relu')(output)
prediction = Dense(1,activation='sigmoid')(output)

model = Model(inputs = mobilenet.input,outputs = prediction)
model.summary()

# Compiling the models
# tell the model what cost and optimization method to use
model.compile(
  loss='binary_crossentropy',
  optimizer=adam_v2(
    learning_rate=0.000001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
),
  metrics=['accuracy']
)

# Setting our model checkpoints
import os
os.mkdir('/content/model_weights/')

model_checkpoint = ModelCheckpoint('./model_weights/antispoofing_model_{epoch:02d}-{val_accuracy:.6f}.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True,save_weights_only=True)

history = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // 25,
    validation_data = valid_generator,
    validation_steps = valid_generator.samples // 25,
    epochs = 100,
    callbacks=[model_checkpoint])

# serialize model to JSON
model_json = model.to_json()
with open("antispoofing_model_mobilenet.json", "w") as json_file:
    json_file.write(model_json)

from keras.preprocessing import image
import numpy as np

# check fakes function
def check_fakes(path,category_type):
  predictor = {}
  for img in os.listdir(path):
    try:
      img = image.load_img(os.path.join(path,img),target_size=(160,160))
      img = image.img_to_array(img)
      img = np.expand_dims(img,axis=0)
      img = img / 255.0
      prediction = model.predict(img)
      if prediction > 0.5:
        prediction_class = 1
      else:
        prediction_class = 0
      result = categories[prediction_class]
      if result not in predictor:
        predictor[result] = 1
      else:
        predictor[result] += 1
    except Exception as e:
      pass
  return predictor