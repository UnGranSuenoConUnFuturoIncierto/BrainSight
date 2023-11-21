import os
import numpy as np
import keras
import tensorflow as tf
import dataLoader
from ConvUNet import ConvUnet

# def diceLoss(yTrue, yPred):
#     flatYTrue = tf.reshape(yTrue, [-1])
#     flatYPred = tf.reshape(yPred, [-1])

#     sum = 0
#     for i in range(len(flatYTrue)):


#     return 

trainImgsPath = "AIsrc/dataForAI/allImagesTrain/"
trainImgsDirList = os.listdir(trainImgsPath)

trainMasksPath = "AIsrc/dataForAI/allMasksTrain/"
trainMasksDirList = os.listdir(trainMasksPath)


valImgsPath = "AIsrc/dataForAI/allImagesValidation/"
valImgsDirList = os.listdir(valImgsPath)

valMasksPath = "AIsrc/dataForAI/allMasksValidation/"
valMasksDirList = os.listdir(valMasksPath)

batchSize = 2
trainImgsLoaderObj = dataLoader.imgsLoader(trainImgsPath, trainImgsDirList, trainMasksPath, trainMasksDirList, batchSize)
valImgsLoaderObj = dataLoader.imgsLoader(valImgsPath, valImgsDirList, valMasksPath, valMasksDirList, batchSize)

# diceLossFunc = diceLoss

metrics = ["accuracy", tf.keras.metrics.MeanIoU(num_classes=4)]
learnRate = 0.0001
optimizer = keras.optimizers.Adam()

stepsEpochTrain = len(trainImgsDirList)//batchSize
stepsEpochValidation = len(valImgsDirList)//batchSize

model = ConvUnet(128,128,128,3,4)
model.compile(optimizer=optimizer, loss="categorical_focal_crossentropy", metrics=metrics)

history = model.fit(trainImgsLoaderObj, steps_per_epoch=stepsEpochTrain, epochs=1, verbose=1, validation_data=valImgsLoaderObj, validation_steps=stepsEpochValidation)

model.save('bratsSeg.hdf5')