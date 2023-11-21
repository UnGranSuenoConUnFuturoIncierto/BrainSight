import os
import numpy as np
import keras
import tensorflow as tf
import dataLoader
from ConvUNet import ConvUnet

trainImgsPath = "AIsrc/dataForAI/allImagesTrain/"
trainImgsDirList = os.listdir(trainImgsPath)

trainMasksPath = "AIsrc/dataForAI/allMasksTrain/"
trainMasksDirList = os.listdir(trainMasksPath)


valImgsPath = "AIsrc/dataForAI/allImagesValidation/"
valImgsDirList = os.listdir(valImgsPath)

valMasksPath = "AIsrc/dataForAI/allMasksValidation/"
valMasksDirList = os.listdir(valMasksPath)

 

def createAndTrain(batchSize=2):

    trainImgsLoaderObj = dataLoader.imgsLoader(trainImgsPath, trainImgsDirList, trainMasksPath, trainMasksDirList, batchSize)
    valImgsLoaderObj = dataLoader.imgsLoader(valImgsPath, valImgsDirList, valMasksPath, valMasksDirList, batchSize)

    metrics = ["accuracy", tf.keras.metrics.MeanIoU(num_classes=4)]

    optimizer = keras.optimizers.Adam()

    stepsEpochTrain = len(trainImgsDirList)//batchSize
    stepsEpochValidation = len(valImgsDirList)//batchSize

    model = ConvUnet(128,128,128,3,4)
    model.compile(optimizer=optimizer, loss="categorical_focal_crossentropy", metrics=metrics)

    history = model.fit(trainImgsLoaderObj, steps_per_epoch=stepsEpochTrain, epochs=1, verbose=1, validation_data=valImgsLoaderObj, validation_steps=stepsEpochValidation)

    model.save('bratsSeg.keras')

def loadAndTrain(modelFile,batchSize=2):

    trainImgsLoaderObj = dataLoader.imgsLoader(trainImgsPath, trainImgsDirList, trainMasksPath, trainMasksDirList, batchSize)
    valImgsLoaderObj = dataLoader.imgsLoader(valImgsPath, valImgsDirList, valMasksPath, valMasksDirList, batchSize)

    model = keras.models.load_model(modelFile)

    stepsEpochTrain = len(trainImgsDirList)//batchSize
    stepsEpochValidation = len(valImgsDirList)//batchSize

    history = model.fit(trainImgsLoaderObj, steps_per_epoch=stepsEpochTrain, epochs=1, verbose=1, validation_data=valImgsLoaderObj, validation_steps=stepsEpochValidation)

    model.save('bratsSeg.keras')

if __name__ == "__main__":
    createAndTrain(12)