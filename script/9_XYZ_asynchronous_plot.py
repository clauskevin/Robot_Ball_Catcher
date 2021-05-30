'''
Dit script zet de opgenomen informatie om in X,Y,Z ruimtecoördinaten.
Vervolgens wordt de opgenomen informatie gevisualiseerd in een 2d plot en een 3d scatterplot.
'''
import matplotlib.pyplot as plt
import numpy
from src import Robot_Ball_Catcher_Functions as func
import os

#gegevens van hsv kalibratie importeren uit de datafolder
datapath=os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'\data'
#intrinsieke en extrinsieke gegevens importeren
mtx=numpy.load(datapath+'\intrinsics.npy')
dist=numpy.load(datapath+'\distortion.npy')
ext=numpy.load(datapath+'\extrinsics.npy')
input=numpy.load(datapath+'\prerecording.npy')

#print de gegevens van de opname
print('input:\n',input)
length = round(len(input[:,:]/4))
print('frames:\n',length)

#maak een nieuwe matrix aan voor de XYZ informatie
worldmatrix=numpy.zeros((length,4))

#overloop de dataset en transformeer deze naar XYZ, plaats deze vervolgens in de worldmatrix
for i in range(length):
    time=input[i,0]
    u_ball=input[i,1]
    v_ball=input[i,2]
    z_ball=input[i,3]
    if 1:
        xcam, ycam, zcam = func.intrinsictrans((u_ball,v_ball), z_ball, mtx)
        xworld, yworld, zworld = func.extrinsictrans(z_ball, xcam, ycam, zcam, ext)
        worldmatrix[i,0]=time
        worldmatrix[i,1]=xworld
        worldmatrix[i,2]=yworld
        worldmatrix[i,3]=zworld

#print deze ruimteinformatie
print('worldmatrix:\n',worldmatrix)
numpy.save(datapath+'\worldmatrix.npy',worldmatrix)

#plotstijl
plt.style.use('dark_background')
#figuur1 is de 2D plot met 3 subplots: X, Y en Z in functie van tijd
#figuur 2 is de 3D XYZ scatterplot
fig1=plt.figure()
fig2=plt.figure()
ax1=fig1.add_subplot(311)
ax2=fig1.add_subplot(312)
ax3=fig1.add_subplot(313)
ax4=fig2.add_subplot(projection='3d')
ax1.plot(worldmatrix[:,0],worldmatrix[:,1])
ax2.plot(worldmatrix[:,0],worldmatrix[:,2])
ax3.plot(worldmatrix[:,0],worldmatrix[:,3])
ax4.scatter(worldmatrix[:,2],worldmatrix[:,1],worldmatrix[:,3])
#aslabels en grafiektitels toevoegen
ax1.set_xlabel('time (ms)')
ax1.set_ylabel('coordinate (mm)')
ax2.set_xlabel('time (ms)')
ax2.set_ylabel('coordinate (mm)')
ax3.set_xlabel('time (ms)')
ax3.set_ylabel('coordinate (mm)')
ax4.set_xlabel('Y (mm)')
ax4.set_ylabel('X (mm)')
ax4.set_zlabel('Z (mm)')
ax1.set_title('X')
ax2.set_title('Y')
ax3.set_title('Z')
ax4.set_title('3D')

#aslimieten manueel instellen volgens werkveld!
ax1.set_ylim(-1000,1000)
ax2.set_ylim(-1000,1000)
ax3.set_ylim(0,600)
#grenzen omwisselen voor juiste conventie met assenstelsel
ax4.set_xlim(1000,-1000)
ax4.set_ylim(-1000,1000)
ax4.set_zlim(0,600)

#automatische opmaak
fig1.tight_layout()
#laat plot zien
plt.show()