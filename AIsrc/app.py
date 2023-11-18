import numpy as np
import nibabel as nib
import cv2
from mayavi import mlab
from sklearn.preprocessing  import MinMaxScaler
import vtk

def getMRIScan(imageNum):
    scaler = MinMaxScaler()
    
    path = f"BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_{imageNum}"

    flair = nib.load(path + f"\BraTS20_Training_{imageNum}_flair.nii").get_fdata()
    flair = scaler.fit_transform(flair.reshape(-1, flair.shape[-1])).reshape(flair.shape)

    t1ce = nib.load(path + f"\BraTS20_Training_{imageNum}_t1ce.nii").get_fdata()
    t1ce = scaler.fit_transform(t1ce.reshape(-1, t1ce.shape[-1])).reshape(t1ce.shape)

    t1 = nib.load(path + f"\BraTS20_Training_{imageNum}_t1.nii").get_fdata()
    t1 = scaler.fit_transform(t1.reshape(-1, t1.shape[-1])).reshape(t1.shape)

    t2 = nib.load(path +f"\BraTS20_Training_{imageNum}_t2.nii").get_fdata()
    t2 = scaler.fit_transform(t2.reshape(-1, t2.shape[-1])).reshape(t2.shape)

    test_mask=nib.load(path + f"\BraTS20_Training_{imageNum}_seg.nii").get_fdata()
    test_mask=test_mask.astype(np.uint8)
    test_mask[test_mask>=1] = 255

    return flair, t1, t1ce, t2, test_mask


def getBrainContours(image):
    rows, cols, floors = image.shape
    onlyContours = np.zeros((rows, cols, floors), dtype="uint8")

    for i in range(floors):
        ithFloor = image[:, :, i].copy()
        ithFloor = ithFloor*255
        ithFloor = np.around(ithFloor)
        ithFloor = ithFloor.astype('uint8')

        cannyImg = cv2.Canny(ithFloor, 100, 120)
        cannyImg = cv2.dilate(cannyImg, None)
        contours, hierarchy = cv2.findContours(cannyImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        

        contourImg = np.zeros((rows, cols))

        cv2.drawContours(contourImg, contours, -1, (255), 1)

        onlyContours[:, :, i] = contourImg

        cv2.imshow("o", image[:, :, i])
        cv2.imshow("!", onlyContours[:, :, i])
        cv2.waitKey(100)

    return onlyContours


flair, t1, t1ce, t2, test_mask = getMRIScan("001")
brainEdge  = getBrainContours(t1)

data = brainEdge + test_mask

# imageData = vtk.vtkImageData()
# imageData.SetDimensions(data.shape[0], data.shape[1], data.shape[2])
# imageData.SetOrigin(0, 0, 0)

# dataArray = vtk.vtkFloatArray()
# dataArray.SetVoidArray(data, 1)
# imageData.SetPointData(dataArray.GetPointer())


# writer = vtk.vtkFileWriter()
# writer.SetFileName("scalarData.vtk")
# writer.Write()

fig = mlab.figure(size=(800, 800))
src = mlab.pipeline.scalar_field(data)
vol = mlab.pipeline.volume(src)




mlab.colorbar()
mlab.outline()
mlab.show()