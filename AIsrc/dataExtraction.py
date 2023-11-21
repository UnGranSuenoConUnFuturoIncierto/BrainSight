import numpy as np
import nibabel as nib
from sklearn.preprocessing  import MinMaxScaler
from keras.utils import to_categorical
import glob

def getTrainData(imageNum):
    scaler = MinMaxScaler()
    
    path = f"AIsrc/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_{imageNum}"

    flair = nib.load(path + f"/BraTS20_Training_{imageNum}_flair.nii").get_fdata()
    flair = scaler.fit_transform(flair.reshape(-1, flair.shape[-1])).reshape(flair.shape)

    t1 = nib.load(path + f"/BraTS20_Training_{imageNum}_t1.nii").get_fdata()
    t1 = scaler.fit_transform(t1.reshape(-1, t1.shape[-1])).reshape(t1.shape)  

    t1ce = nib.load(path + f"/BraTS20_Training_{imageNum}_t1ce.nii").get_fdata()
    t1ce = scaler.fit_transform(t1ce.reshape(-1, t1ce.shape[-1])).reshape(t1ce.shape)

    t2 = nib.load(path +f"/BraTS20_Training_{imageNum}_t2.nii").get_fdata()
    t2 = scaler.fit_transform(t2.reshape(-1, t2.shape[-1])).reshape(t2.shape)

    mask = nib.load(path + f"/BraTS20_Training_{imageNum}_seg.nii").get_fdata()
    mask = mask.astype(np.uint8)
    mask[mask==4] = 3

    return flair, t1, t1ce, t2, mask

def getValidationData(imageNum):
    scaler = MinMaxScaler()
    
    path = f"AIsrc/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/BraTS20_Validation_{imageNum}"

    flair = nib.load(path + f"/BraTS20_Validation_{imageNum}_flair.nii").get_fdata()
    flair = scaler.fit_transform(flair.reshape(-1, flair.shape[-1])).reshape(flair.shape)

    t1 = nib.load(path + f"/BraTS20_Validation_{imageNum}_t1.nii").get_fdata()
    t1 = scaler.fit_transform(t1.reshape(-1, t1.shape[-1])).reshape(t1.shape)

    t1ce = nib.load(path + f"/BraTS20_Validation_{imageNum}_t1ce.nii").get_fdata()
    t1ce = scaler.fit_transform(t1ce.reshape(-1, t1ce.shape[-1])).reshape(t1ce.shape)

    t2 = nib.load(path +f"/BraTS20_Validation_{imageNum}_t2.nii").get_fdata()
    t2 = scaler.fit_transform(t2.reshape(-1, t2.shape[-1])).reshape(t2.shape)

    return flair, t1, t1ce, t2

def combineAIData(flair, t1ce, t2):
    return np.stack([flair, t1ce, t2], axis=3)

def cropAIData(aiData):
    return aiData[56:184,56:184, 13:141]

def saveArray(path, mat):
    np.save(path, mat)

def maskToCategorical(mask):
    return to_categorical(mask, num_classes=4)


if __name__ =="__main__":
    import os
    print(os.getcwd())


    t2_list = sorted(glob.glob('AIsrc/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*t2.nii'))
    t1ce_list = sorted(glob.glob('AIsrc/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*t1ce.nii'))
    flair_list = sorted(glob.glob('AIsrc/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*flair.nii'))
    mask_list = sorted(glob.glob('AIsrc/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*seg.nii'))

    for img in range(len(t2_list)):   #Using t1_list as all lists are of same size
        fullPath = t2_list[img]
        numImagePath = fullPath.split("_")[-2]

        flair, t1, t1ce, t2, mask = getTrainData(numImagePath)
        
        combinedAIData = combineAIData(flair,t1ce,t2)
        combinedAIData = cropAIData(combinedAIData)
        
        mask = cropAIData(mask)
        mask = to_categorical(mask, num_classes=4)

        np.save('AIsrc/dataForAI/allImagesTrain/image_'+numImagePath+'.npy', combinedAIData)
        np.save('AIsrc/dataForAI/allMasksTrain/mask_'+numImagePath+'.npy', mask)
