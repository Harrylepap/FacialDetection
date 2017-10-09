import os

import cv2
import numpy as np
from PIL import Image

reconnaissance = cv2.face.LBPHFaceRecognizer_create()
chemin = "dataSet"


def getImageAvecID(chemin):
    imagePaths = [os.path.join(chemin, f) for f in os.listdir(chemin)]
    faces = []
    IDs = []
    for imagePath in imagePaths:
        faceImage = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImage, 'uint8')
        ID = int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow("Entrainement", faceNp)
        cv2.waitKey(10)
    return np.array(IDs), faces


IDs, faces = getImageAvecID(chemin)
reconnaissance.train(faces, IDs)
reconnaissance.write("entrainement/data.yml")
cv2.destroyAllWindows()
