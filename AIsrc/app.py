import numpy as np
import nibabel as nib
import cv2
from sklearn.preprocessing  import MinMaxScaler
from skimage import measure
import trimesh
import matplotlib.pyplot as plt

def getTrainData(imageNum):
    scaler = MinMaxScaler()
    
    path = f"AIsrc\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_{imageNum}"

    flair = nib.load(path + f"\BraTS20_Training_{imageNum}_flair.nii").get_fdata()
    flair = scaler.fit_transform(flair.reshape(-1, flair.shape[-1])).reshape(flair.shape)*255
    flair = flair.astype("uint8")

    t1 = nib.load(path + f"\BraTS20_Training_{imageNum}_t1.nii").get_fdata()
    t1 = scaler.fit_transform(t1.reshape(-1, t1.shape[-1])).reshape(t1.shape)*255
    t1 = t1.astype("uint8")     

    t1ce = nib.load(path + f"\BraTS20_Training_{imageNum}_t1ce.nii").get_fdata()
    t1ce = scaler.fit_transform(t1ce.reshape(-1, t1ce.shape[-1])).reshape(t1ce.shape)*255
    t1ce = t1ce.astype("uint8")   

    t2 = nib.load(path +f"\BraTS20_Training_{imageNum}_t2.nii").get_fdata()
    t2 = scaler.fit_transform(t2.reshape(-1, t2.shape[-1])).reshape(t2.shape)*255
    t2 = t2.astype("uint8")

    mask = nib.load(path + f"\BraTS20_Training_{imageNum}_seg.nii").get_fdata()
    mask = mask.astype(np.uint8)

    return flair, t1, t1ce, t2, mask

def getValidationData(imageNum):
    scaler = MinMaxScaler()
    
    path = f"AIsrc\BraTS2020_ValidationData\MICCAI_BraTS2020_ValidationData\BraTS20_Validation_{imageNum}"

    flair = nib.load(path + f"\BraTS20_Validation_{imageNum}_flair.nii").get_fdata()
    flair = scaler.fit_transform(flair.reshape(-1, flair.shape[-1])).reshape(flair.shape)*255
    flair = flair.astype("uint8")

    t1 = nib.load(path + f"\BraTS20_Validation_{imageNum}_t1.nii").get_fdata()
    t1 = scaler.fit_transform(t1.reshape(-1, t1.shape[-1])).reshape(t1.shape)*255
    t1 = t1.astype("uint8")

    t1ce = nib.load(path + f"\BraTS20_Validation_{imageNum}_t1ce.nii").get_fdata()
    t1ce = scaler.fit_transform(t1ce.reshape(-1, t1ce.shape[-1])).reshape(t1ce.shape)*255
    t1ce = t1ce.astype("uint8")

    t2 = nib.load(path +f"\BraTS20_Validation_{imageNum}_t2.nii").get_fdata()
    t2 = scaler.fit_transform(t2.reshape(-1, t2.shape[-1])).reshape(t2.shape)*255
    t2 = t2.astype("uint8")

    return flair, t1, t1ce, t2


def getBrainContours(image):
    rows, cols, floors = image.shape
    onlyContours = np.zeros((rows, cols, floors), dtype="uint8")

    for i in range(floors):

        cannyImg = cv2.Canny(image[:, :, i], 100, 120)
        cannyImg = cv2.dilate(cannyImg, None)
        contours, hierarchy = cv2.findContours(cannyImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        contourImg = np.zeros((rows, cols))

        cv2.drawContours(contourImg, contours, -1, (255), 1)

        onlyContours[:, :, i] = contourImg

    return onlyContours

def getBooleanMasks(mask):
    onesMask = mask == 1
    onesMask[:, :, -1] = False  
    onesMask[:, :, 0] = False

    twosMask = mask == 2
    twosMask[:, :, -1] = False
    twosMask[:, :, 0] = False
    
    foursMask = mask == 4
    foursMask[:, :, -1] = False
    foursMask[:, :, 0] = False

    return onesMask, twosMask, foursMask

def create3DMesh(mask, vertColors = np.array([0,0,0]), alpha = 255):
    vertices, faces, normals, values = measure.marching_cubes(mask)
    
    rows, cols = faces.shape
    colors = np.zeros((rows, 4))
    colors[:, :3] = vertColors 
    colors[:, 3] = alpha
    mesh = trimesh.Trimesh(vertices = vertices, faces=faces, face_colors=colors)
    return mesh

def create3DBrainWithTumor_Train(file):
    flair, t1, t1ce, t2, mask = getTrainData(file)
    
    brainContour = getBrainContours(t1)

    one, two, four = getBooleanMasks(mask)

    brainContourMesh = create3DMesh(brainContour==255, alpha=70)
    onesMesh = create3DMesh(one, vertColors=np.array([255,0,0]), alpha=150)
    twosMesh = create3DMesh(two, vertColors=np.array([0,255,0]), alpha=100)
    foursMesh = create3DMesh(four, vertColors=np.array([0,0,255]), alpha=115)
    meshes = [onesMesh, foursMesh, twosMesh, brainContourMesh]
    combined = trimesh.util.concatenate(meshes)
    combined.show()
    combined.export("test.obj")
    combined.export("test.glb")
    combined.export("test.stl")

def getFrame(image, i, path):
    cv2.imwrite(path,image[:,:,i])

def getMask(image, i, path):
   plt.imshow(image[:, :, i])
   plt.colorbar()
   plt.savefig(path)



if __name__ =="__main__":
    flair, t1, t1ce, t2, mask = getTrainData("001")
    getMask(mask, 50,"")
    #create3DBrainWithTumor_Train("186")