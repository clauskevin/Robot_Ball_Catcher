"""
Met deze tool wordt de baldetectie gevisualiseerd,
wordt de pixelcoördinaat UV en diepte Z geprojecteerd op het kleurbeeld
en wordt de X,Y,Z ruimtecoördinaat geprojecteerd. (extrinsieke kalibratie moet gebeurd zijn.)
De X,Y,Z ruimtecoördinaat wordt realtime geplot.
"""

#bibliotheken
import numpy
from src import Robot_Ball_Catcher_Functions as func
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

#functie om de plot te vernieuwen
def updatelists(d,x,y,z):
    #er wordt steeds een reeks van 100 waarden gevisualiseerd,
    #wanneer er een nieuwe waarde is moet de eerste waarde weg en schuift alles
    #dan kan er op het einde een nieuwe waarde bij
    #indien geen diepte geregistreerd, maak alles 0
    X.popleft()
    Y.popleft()
    Z.popleft()
    if d:
        X.append(x)
        Y.append(y)
        Z.append(z)
    else:
        X.append(0)
        Y.append(0)
        Z.append(0)

    #maak de assen schoon
    ax1.cla()
    ax2.cla()
    ax3.cla()
    #stel labels en titels in voor de assen
    ax1.set_xlabel('time (ms)')
    ax1.set_ylabel('coordinate (mm)')
    ax2.set_xlabel('time (ms)')
    ax2.set_ylabel('coordinate (mm)')
    ax3.set_xlabel('time (ms)')
    ax3.set_ylabel('coordinate (mm)')
    ax1.set_title('X')
    ax2.set_title('Y')
    ax3.set_title('Z')
    #tel limieten in voor de assen
    ax1.set_ylim(-300, 300)
    ax2.set_ylim(-300, 300)
    ax3.set_ylim(0, 300)
    #visualiseer de nieuwe reeks waarden op de plot
    ax1.plot(X)
    ax2.plot(Y)
    ax3.plot(Z)
    #automatische opmaak
    fig.tight_layout()

#animatiefunctie met de loop van het script in
def updateplot(i):
    #beelden ophalen
    color_image, depth_image = func.getframes(pipeline, align_object)
    #masker van de bal opvragen
    mask=func.getmask(color_image,lower_color,upper_color)
    #pixel van het zwaartepunt van de bal zoekekn
    ball_pixel, radius = func.getballpixel(mask, minradius)
    #wanneer er een bal gevonden is is er een zwaartepunt (pixel), bepaal hier de diepte
    if ball_pixel:
        depth_pixel = func.getdepthpixel(depth_image, ball_pixel)
    else:
        depth_pixel = None
    #intrinsiek transformeren naar cameracoördinaten
    xcam, ycam, zcam = func.intrinsictrans(ball_pixel, depth_pixel, mtx)
    #extrinsiek transformeren naar ruimtecoördinaten
    xworld, yworld, zworld = func.extrinsictrans(depth_pixel, xcam, ycam, zcam, ext)
    #visualiseer de gedetecteerde bal en de UV en Z waarde op het kleurenbeeld
    img = func.showballpixel(color_image, ball_pixel, depth_pixel, xworld, yworld, zworld, radius)
    img = func.drawaxes(img, mtx, dist, rvecs, tvecs, 60)
    #beelden laten zien
    func.showimage(img, window_name)
    #plot de waarden
    updatelists(depth_pixel, xworld, yworld, zworld)

#gegevens van hsv kalibratie importeren uit de datafolder
datapath=os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'\data'
hsvfile=numpy.load(datapath+'\hsv.npy')
# onderste en bovenste kleur definieren
lower_color = numpy.array([hsvfile[0],hsvfile[2],hsvfile[4]])
upper_color = numpy.array([hsvfile[1],hsvfile[3],hsvfile[5]])

#intrinsieke en extrinsieke gegevens importeren
mtx=numpy.load(datapath+'\intrinsics.npy')
dist=numpy.load(datapath+'\distortion.npy')
ext=numpy.load(datapath+'\extrinsics.npy')
rvecs=numpy.load(datapath+'\extrinsic_rvecs.npy')
tvecs=numpy.load(datapath+'\extrinsic_tvecs.npy')

#venster instellingen
window_name='XYZ realtime plot'
window_b, window_h = 1920, 1080

#camera instellingen
color_resolution = (1920,1080)
depth_resolution = (1280,720)
frames_per_second = 30

#minimum straal van de bal contour
minradius=30

#stel een plot in met 3 verticale subplots
fig=plt.figure()
ax1=plt.subplot(311)
ax2=plt.subplot(312)
ax3=plt.subplot(313)

#maak de lege lijsten die gevisualiseerd worden op de plot
X=deque(numpy.zeros(100))
Y=deque(numpy.zeros(100))
Z=deque(numpy.zeros(100))

#stel venster en camera in
func.setwindow(window_name,window_b,window_h)
pipeline, align_object = func.startcamera(color_resolution,depth_resolution,frames_per_second)

#plot animatiefunctie die steeds de updateplot functie zal herhalen als loop
ani=FuncAnimation(fig,updateplot,interval=1)
#voer het plot uit
plt.show()