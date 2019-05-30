import numpy as np
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

filepath = './img/me.jpg'

base_model = VGG16()
base_model.summary()
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block2_conv2').output)

img = image.load_img(filepath, target_size=(224, 224))
x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

interlayer_output = model.predict(x)
print(interlayer_output.shape)
for ind in range(16):
    image = interlayer_output[0,:,:,ind]
    plt.subplot(4,4,ind+1)
    plt.imshow(image)
    # print(image.shape)
plt.show()


interlayer_weights = base_model.get_layer('block2_conv2').get_weights()
print(interlayer_weights[0].shape)