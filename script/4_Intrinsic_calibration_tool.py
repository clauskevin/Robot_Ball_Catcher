'''
Met dit script kan de intrinsieke matrix van de camera bepaald worden
'''
#bibliotheken
import cv2 as opencv
import numpy
from src import Robot_Ball_Catcher_Functions as func

#eigenschappen patroon
h=14
b=9
size=17.6 #aantal mm zijde van een vierkant op het schaakbord

#aantal kalibratiewaarden
amount=30

#venster instellingen
window_name='Intrinsic calibration'
window_b, window_h = 1920, 1080

#camera instellingen
color_resolution = (1920,1080)
depth_resolution = (1280,720)
frames_per_second = 30

#stel venster en camera in
func.setwindow(window_name,window_b,window_h)
pipeline, align_object = func.startcamera(color_resolution,depth_resolution,frames_per_second)

# termination criteria max 30 iteraties of epsilon(nauwkeurigheid) van 0,001
criteria = (opencv.TERM_CRITERIA_EPS + opencv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# bereid de objectpunten van het schaakbordpatroon voor
objp = numpy.zeros((h*b,3), numpy.float32)
objp[:,:2] = numpy.mgrid[0:b,0:h].T.reshape(-1,2)
objp = size*objp #werkelijke grootte

#wereldpunten en beeldpunten verzamelen in arrays
objpoints = [] # 3d punten in de ruimte
imgpoints = [] # corresponderende punten in 2d

#voer de kalibratie uit tot er 'amount' aantal waarden zijn
k=0
while k<amount:
    #max 1 waarde per seconde registreren
    opencv.waitKey(1000)
    #beelden ophalen
    img, depth_image = func.getframes(pipeline, align_object)
    #grijsschaal beeld maken zodat schaakbordpatroon beter herkend wordt
    gray = opencv.cvtColor(img, opencv.COLOR_BGR2GRAY)
    #zoek hoekpunten op schaakbord
    ret, corners = opencv.findChessboardCorners(gray, (b, h), None)
    #indien hoekpunten, zoek verfijnde subpixel hoekpunten en vul de 2d en 3d lijsten
    if ret == True:
        objpoints.append(objp)
        corners2 = opencv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        #visualiseer de hoekpunten
        img = opencv.drawChessboardCorners(img, (b, h), corners2, ret)
        func.showimage(img, window_name)
        #doe de k teller omhoog want er is een kalibratiewaarde geregistreerd
        k=k+1
        #print hoeveel meetwaarden er al zijn
        print(k,"\n")
    else:
        func.showimage(img, window_name)

#kalibratie afgelopen, sluit de vensters
opencv.destroyAllWindows()

#bereken de intrinsieke eigenschappen van de camera
ret, mtx, dist, rvecs, tvecs = opencv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print("intrinsieke matrix:\n")
print(mtx)
print("distortie:\n")
print(dist)

#bereken de fout, deze moet kleiner zijn dan 1 voor een geslaagde kalibratie, hoe kleiner hoe beter
mean_error =0
for i in range(len(objpoints)):
    imgpoints2, _ = opencv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = opencv.norm(imgpoints[i],imgpoints2, opencv.NORM_L2)/len(imgpoints2)
    mean_error += error
#print de fout:
print("\ntotal error:\n", mean_error/len(objpoints))
'''
opslaan in array niet uitgewerkt omdat dit project werkt met de D415.
de intrinsieke matrix hiervan werd bepaald met de fabrieksgegevens:
intrinsieke matrix:
[[1.34e+03 0.00e+00 9.60e+02]
 [0.00e+00 1.34e+03 5.40e+02]
 [0.00e+00 0.00e+00 1.00e+00]]
distorsie verwaarloosbaar:
[0. 0. 0. 0. 0.]
dit script komt ongeveer dezelfde waarden uit
'''
