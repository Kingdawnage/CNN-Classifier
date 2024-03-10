from tabnanny import verbose
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array
from keras.models import load_model

import os

# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# train_horse_dir = os.path.join('./assets/training/horses')
# train_human_dir = os.path.join('./assets/training/humans')

# train_horse_names = os.listdir(train_horse_dir)
# train_human_names = os.listdir(train_human_dir)

# nrows = 4
# ncols = 4

# # Index for iterating over images
# pic_index = 0
# # Set up matplotlib fig, and size it to fit 4x4 pics
# fig = plt.gcf()
# fig.set_size_inches(ncols * 4, nrows * 4)

# pic_index += 8
# next_horse_pix = [os.path.join(train_horse_dir, fname) 
#                 for fname in train_horse_names[pic_index-8:pic_index]]
# next_human_pix = [os.path.join(train_human_dir, fname) 
#                 for fname in train_human_names[pic_index-8:pic_index]]

# for i, img_path in enumerate(next_horse_pix+next_human_pix):
#   # Set up subplot; subplot indices start at 1
#   sp = plt.subplot(nrows, ncols, i + 1)
#   sp.axis('Off') # Don't show axes (or gridlines)

#   img = mpimg.imread(img_path)
#   plt.imshow(img)

# #plt.show()

# model = tf.keras.models.Sequential([
#         # First conv layer
#         tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
#         tf.keras.layers.MaxPooling2D(2,2),
#         # Second conv layer
#         tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
#         tf.keras.layers.MaxPooling2D(2,2),
#         # Third conv layer
#         tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#         tf.keras.layers.MaxPooling2D(2,2),
#         # Fourth conv layer
#         tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#         tf.keras.layers.MaxPooling2D(2,2),
#         # Fifth conv layer
#         tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
#         tf.keras.layers.MaxPooling2D(2,2),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(512, activation='relu'),
#         tf.keras.layers.Dense(1, activation='sigmoid')
#         ])
# model.summary()
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # All images will be rescaled by 1./255
# train_datagen = ImageDataGenerator(rescale=1/255)
# validation_datagen = ImageDataGenerator(rescale=1/255)

# # Flow training images in batches of 128 using train_datagen generator
# train_generator = train_datagen.flow_from_directory(
#         './assets/training/',  # This is the source directory for training images
#         target_size=(300, 300),  # All images will be resized to 300x300
#         batch_size=128,
#         # Since you use binary_crossentropy loss, you need binary labels
#         class_mode='binary')

# # Flow validation images in batches of 128 using validation_datagen generator
# validation_generator = validation_datagen.flow_from_directory(
#         './assets/validation/',  # This is the source directory for validation images
#         target_size=(300, 300),  # All images will be resized to 300x300
#         batch_size=32,
#         # Since you use binary_crossentropy loss, you need binary labels
#         class_mode='binary')

# history = model.fit(train_generator,
#                     steps_per_epoch=8,
#                     epochs=15,
#                     verbose=1,
#                     validation_data=validation_generator,
#                     validation_steps=8)

## Uncomment this to save a model you want to train. Remember to uncomment the code above to train your own model
## if you want to train the model for more classes
# model.save(os.path.join('models', 'two_class_classifier.h5'))

loaded_model = load_model('./models/two_class_classifier.h5')

# All you need to do is add a picture the prediction folder to make a prediction on it
predict_img = [os.path.join('./assets/prediction', f) for f in os.listdir('./assets/prediction') if f.endswith(('.jpg', '.jpeg', '.png'))]

for image_path in predict_img:
    img = load_img(image_path, target_size=(300, 300))
    x = img_to_array(img)
    x /= 255
    x = np.expand_dims(x, axis=0)

    # Predict class probabilities
    ## This code uses the presave model to make predictions. Comment it out and use the line below to use the model you trained
    ## You can use model you saved and trained as well
    classes = loaded_model.predict(x)
    ## Uncomment this code to train the model you have trained
    #classes = model.predict(x)
    # Print the predicted class probabilities
    print(classes[0])

    # Determine the predicted class
    if classes[0] > 0.5:
        print(os.path.basename(image_path) + " is a human")
    else:
        print(os.path.basename(image_path) + " is a horse")