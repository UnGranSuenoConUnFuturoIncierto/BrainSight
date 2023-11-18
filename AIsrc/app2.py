import numpy as np
import nibabel as nib
import cv2
from mayavi import mlab
from sklearn.preprocessing  import MinMaxScaler

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
    test_mask[test_mask==4] = 3

    return flair, t1, t1ce, t2, test_mask


#cv2.imshow("idk", flair[:, :, 30])
#cv2.waitKey(0)


#FLAIR ES EL FLUIDO DEL CEREBRO
#T1 ES TEJIDO
#T2 ES FLUIDOS
#T1CE ES CON CONTRASTE DE ALGO QUE NO SABEMOS

#HACER EL CONTORNO DEL CEREBRO CON LA MATRIZ



#HACER LA IMAGEN 3D DE SOLO TUMOR CON SOLO CONTRASTE
flair, t1, t1ce, t2, test_mask = getMRIScan("001")

data = t1

fig = mlab.figure(size=(800, 800))
src = mlab.pipeline.scalar_field(data)
vol = mlab.pipeline.volume(src)

mlab.colorbar(title='Tumor solo')
mlab.outline()
mlab.show()


#1. SACAR SOLO EL TUMOR DE LA MATRIZ
#2. HACER UNA MATRIZ CON SOLO TUMOR Y EL CONTORNO DEL CEREBRO