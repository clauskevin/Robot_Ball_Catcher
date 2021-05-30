'''
Dit script dient uitgevoerd te worden na de asynchrone visualisatie.
Er wordt een regressie uitgevoerd op een vooraf gedefinieerd bereik aan waarden.
Het regressieverloop doorheen de frames wordt bij uitvoeren van het script gevisualiseerd.
'''

#bibliotheken
import matplotlib.pyplot as plt
import numpy
from scipy import optimize
import time
import os

#definieer het aantal waarden van de opname die horen bij de worp:
frames=13


#regressiefuncties definieren
def test_func_z(t,a,b,c):
    return a*t+b*t*t+c
def test_func_xy(t,a,b):
    return a*t+b
#interactief plotten activeren
plt.ion()
#plotstijl
plt.style.use('dark_background')
#figuur 1: X, Y en Z in functie van tijd 2D subplots
#figuur 2: 3D scatterplot XYZ
fig1 = plt.figure()
fig2 = plt.figure()
ax1 = fig1.add_subplot(311)
ax2 = fig1.add_subplot(312)
ax3 = fig1.add_subplot(313)
ax4 = fig2.add_subplot(projection = '3d')


datapath=os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'\data'
input=numpy.load(datapath+'\worldmatrix.npy')
worldmatrix=numpy.array(input[:frames-1,:])

#print dataset
print('input:\n',worldmatrix)
length = round(len(worldmatrix[:,:]/4))
print('frames:\n',length)

k=3 #regressie start vanaf 3 waarden, vroeger kan niet aangezien er kwadratische regressie is op Z

#regressie simulatie, steeds 1 frame meer toevoegen aan de regressie dataset
while k<frames+1:
    print(k)
    #reset k voor een loop
    if k==frames:
        k=3
    #regressie
    paramsx, params_covariancex = optimize.curve_fit(test_func_xy, worldmatrix[:k, 0], worldmatrix[:k, 1])
    paramsy, params_covariancey = optimize.curve_fit(test_func_xy, worldmatrix[:k, 0], worldmatrix[:k, 2])
    paramsz, params_covariancez = optimize.curve_fit(test_func_z, worldmatrix[:k, 0], worldmatrix[:k, 3])
    #plots schoonmaken
    ax1.cla()
    ax2.cla()
    ax3.cla()
    ax4.cla()
    #labelen van assen + grafiektitels
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

    # aslimieten manueel instellen volgens werkveld!
    ax1.set_ylim(-1000, 1000)
    ax2.set_ylim(-1000, 1000)
    ax3.set_ylim(0, 600)
    # grenzen omwisselen voor juiste conventie met assenstelsel
    ax4.set_xlim(1000, -1000)
    ax4.set_ylim(-1000, 1000)
    ax4.set_zlim(0, 600)


    #plotten van de informatie
    ax1.scatter(worldmatrix[:, 0], worldmatrix[:, 1])
    ax2.scatter(worldmatrix[:, 0], worldmatrix[:, 2])
    ax3.scatter(worldmatrix[:, 0], worldmatrix[:, 3])
    ax4.scatter(worldmatrix[:, 2], worldmatrix[:, 1], worldmatrix[:, 3])
    ax1.plot(worldmatrix[:, 0], test_func_xy(worldmatrix[:, 0], paramsx[0], paramsx[1]))
    ax2.plot(worldmatrix[:, 0], test_func_xy(worldmatrix[:, 0], paramsy[0], paramsy[1]))
    ax3.plot(worldmatrix[:, 0], test_func_z(worldmatrix[:, 0], paramsz[0], paramsz[1], paramsz[2]))
    ax4.plot(test_func_xy(worldmatrix[:, 0], paramsy[0], paramsy[1]),\
             test_func_xy(worldmatrix[:, 0], paramsx[0], paramsx[1]),\
             test_func_z(worldmatrix[:, 0], paramsz[0], paramsz[1], paramsz[2]))
    fig1.canvas.draw()
    fig2.canvas.draw()
    fig1.canvas.flush_events()
    fig2.canvas.flush_events()
    #halve seconde tussen iedere visualisatie
    time.sleep(0.5)
    #k telt op
    k=k+1
