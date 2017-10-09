import sqlite3

import cv2
from notebook.notebookapp import raw_input


def ajoutUpdate(id, nom):
    connection = sqlite3.connect("facebase.db")
    requete = "SELECT * FROM personne WHERE id =" + str(id)
    datas = connection.execute(requete)
    personneExiste = 0
    for data in datas:
        personneExiste = 1
    if (personneExiste == 1):
        requete = "UPDATE personne SET nom=" + str(nom) + " WHERE id=" + str(id)
    else:
        requete = "INSERT INTO personne(id, nom) VALUES(" + str(id) + ", " + str(nom) + ")"
    connection.execute(requete)
    connection.commit()
    connection.close()


faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
camera = cv2.VideoCapture(0)
id = raw_input("Utilisateur ID :")
nom = raw_input("Utilisateur nom :")
ajoutUpdate(id, nom)
sampleNum = 0
while (True):
    ret, image = camera.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        sampleNum += 1
        cv2.imwrite("dataSet/Utilisateur." + str(id) + "." + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.waitKey(100)
    cv2.imshow("Tarehy", image)
    cv2.waitKey(1)
    if (sampleNum > 20):
        break
camera.release()
cv2.destroyAllWindows()
