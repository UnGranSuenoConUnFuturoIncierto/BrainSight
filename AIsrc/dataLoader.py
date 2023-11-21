import numpy as np
import os

def imgsLoad(path, imgsDirList):
    imgs = []
    for imgPath in imgsDirList:
        imgMat = np.load(path+imgPath)
        imgs.append(imgMat)
    
    imgs = np.array(imgs)
    return imgs

def imgsLoader(imgsPath, imgsDirList, masksPath, masksDirList, batchSize):
    while True:
        i=0
        while i < (len(imgsDirList)):    
            imgs = imgsLoad(imgsPath, imgsDirList[i: i + batchSize])
            masks = imgsLoad(masksPath, masksDirList[i: i + batchSize])

            yield (imgs, masks)

            i+=batchSize


if __name__ == "__main__":
    imgsPath = "AIsrc/dataForAI/allImagesValidation/"
    masksPath = "AIsrc/dataForAI/allMasksTrain/"
    imgsDirList = os.listdir(imgsPath)
    masksDirList = os.listdir(masksPath)
    
    imgsLoaderObj = imgsLoader(imgsPath, imgsDirList, masksPath, masksDirList, 1)
    
    i = 1
    while True:
        imgs, masks = imgsLoaderObj.__next__()
        print(i)
        print(np.unique(imgs))
        input()
        i+=1