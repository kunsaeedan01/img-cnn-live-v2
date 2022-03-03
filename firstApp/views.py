from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
from tensorflow import Graph
import json
import numpy as np
from firstApp.models import ClassifiedImages


img_height, img_width = 224, 224
with open('./models/imagenet_classes.json', 'r') as f:
    labelInfo = f.read()

labelInfo = json.loads(labelInfo)
model = load_model('./models/MobileNetModelImagenet.h5')


model_graph = Graph()
with model_graph.as_default():
    tf_session = tf.compat.v1.Session()
    with tf_session.as_default():
        model = load_model('./models/MobileNetModelImagenet.h5')


def index(request):
    context = {'a':1}
    return render(request, 'index.html', context)


def predictImage(request):
    fileObj = request.FILES['filePath']
    fs = FileSystemStorage()
    filePathName = fs.save(fileObj.name, fileObj)
    filePathName = fs.url(filePathName)
    testimage = '.' + filePathName
    img = image.load_img(testimage, target_size=(img_height, img_width))
    x = image.img_to_array(img)
    x = x / 255
    x = x.reshape(1, img_height, img_width, 3)
    with model_graph.as_default():
        with tf_session.as_default():
            prediction = model.predict(x)

    predictedLabel = labelInfo[str(np.argmax(prediction[0]))]

    new_entry = ClassifiedImages(link=testimage, classification=predictedLabel[1])
    new_entry.save()

    context = {'filePathName': filePathName, 'predictedLabel': predictedLabel[1]}
    return render(request, 'index.html', context)