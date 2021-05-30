#bibliotheken
import pyrealsense2 as realsense
import numpy
import cv2 as opencv

#camera instellen
def startcamera(cres, dres,fps):
    #verbinden met de camera
    connection = realsense.pipeline()
    #configureren van instellingen
    configuration = realsense.config()
    configuration.enable_stream(realsense.stream.depth, dres[0], dres[1], realsense.format.z16, fps)
    configuration.enable_stream(realsense.stream.color, cres[0], cres[1], realsense.format.bgr8, fps)
    #verbinding starten
    connection.start(configuration)
    #aligneren van beelden instellen
    align = realsense.align(realsense.stream.color)
    return connection, align

#beelden ophalen
def getframes(connection, align):
    #wachten op een beschikbaar beeld
    frames = connection.wait_for_frames()
    #beelden aligneren
    aligned_frames = align.process(frames)
    #beelden ophalen
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    #beelden omzetten in arrays
    depth = numpy.asanyarray(depth_frame.get_data())
    color = numpy.asanyarray(color_frame.get_data())
    return color, depth

def getmask(image,mincolor, maxcolor):
    #camerabeeld wazig maken om details weg tewerken
    blurred = opencv.GaussianBlur(image, (11, 11), 0)
    #omzetten naar hsv kleurruimte
    hsv = opencv.cvtColor(blurred, opencv.COLOR_BGR2HSV)
    #masker uitwerken
    mask = opencv.inRange(hsv, mincolor, maxcolor)
    #eroderen en verwijden om masker te verfijnen
    mask = opencv.erode(mask, None, iterations = 2)
    mask = opencv.dilate(mask, None, iterations = 2)
    return mask

#vraag de pixel op die het center van de bal voorstelt in het masker
def getballpixel(mask,r):
    #zoek contouren in het masker
    contours, hierarchy = opencv.findContours(mask, opencv.RETR_EXTERNAL, opencv.CHAIN_APPROX_SIMPLE)
    #als er contouren zijn kan het de bal zijn, anders niet
    if len(contours) > 0:
        #zoek de contour met de grootste oppervlakte
        maxcontour = max(contours, key=opencv.contourArea)
        #zoek de momenten van de grootste contour
        moments = opencv.moments(maxcontour)
        #zoek het center van de grootste contour met de momentenformule indien mogelijk
        try:
            center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
        except:
            center=None
        #zoek de straal van de minimum omsloten circel
        ((x, y), radius) = opencv.minEnclosingCircle(maxcontour)
        #als de straal groot genoeg is gaat het om de bal, anders niet
        if radius > r:
            return center,radius
        else:
            return None, None
    else:
        return None, None

#vraag de dieptewaarde op van een bepaalde pixel in dieptebeeld
def getdepthpixel(image,pixel):
    depth = image[pixel[1],pixel[0]]
    return depth

#visualisatie van gedetecteerde bal, balpixel, diepte en XYZ coördinaat indien van toepassing
def showballpixel(image,pixel,depth,x,y,z,r):
    #als er een bal is, visualiseer de bal aan de hand van een stip in het midden en een omsloten cirkel
    if pixel:
        opencv.circle(image, pixel, 5, (0, 0, 255), -1)
        opencv.circle(image, pixel, int(r), (255,0,0),5)
    center_as_string = ''.join(str(pixel))
    #bij een dieptewaarde van 0 is de bal te dicht, verander de geprojecteerde waarde naar 'TO CLOSE'
    if depth==0:
        depth_as_string='TO CLOSE'
    else:
        depth_as_string = str(depth)
    #projecteer waarde voor ballpixel en diepte
    opencv.putText(image, "pixel: "+center_as_string, (10, 300), opencv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, opencv.LINE_AA)
    opencv.putText(image, "depth in mm: "+depth_as_string, (10, 400), opencv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, opencv.LINE_AA)
    #Indien er ruimtecoördinaatgegevens zijn van de bal, projecteer ze
    if depth:
        if x or y or z:
            opencv.putText(image, "X: " + str(int(round(x)))+' mm', (10, 500), opencv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255),2, opencv.LINE_AA)
            opencv.putText(image, "Y: " + str(int(round(y)))+' mm', (10, 600), opencv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255),2, opencv.LINE_AA)
            opencv.putText(image, "Z: " + str(int(round(z)))+' mm', (10, 700), opencv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255),2, opencv.LINE_AA)
    return image

def intrinsictrans(pixel,z,mtx):
    if(z):
        x=(pixel[0]-mtx[0,2])/mtx[0,0]*z
        y=(pixel[1]-mtx[1,2])/mtx[1,1]*z
        return x,y,z
    else:
        return None,None,None

def extrinsictrans(depth, x,y,z,ext):
    if(depth):
        mat = numpy.array([[x],[y],[z],[1]])
        inv =numpy.linalg.inv(ext)
        world = numpy.dot(inv, mat)
        xw, yw, zw = world[0,0], world[1,0], world[2,0],
        newx=yw
        newy=xw
        newz=-zw
        return newx, newy, newz
    else:
        return None, None, None

#teken de assen van het referentiestelsel
def drawaxes(img,mtx,dist,rvecs,tvecs,length):
    #vectors van de assen volgens 3D
    axis = numpy.float32([[0,0,0],[length, 0, 0], [0, length, 0], [0, 0, -length]]).reshape(-1, 3)
    #deze punten afbeelden volgens intrinsieke en extrinsieke eigenschappen op het 2d beeld
    imgpts,jac=opencv.projectPoints(axis, rvecs, tvecs, mtx, dist)
    #projectie van de lijnen
    corner = tuple(imgpts[0].ravel())
    img = opencv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = opencv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    img = opencv.line(img, corner, tuple(imgpts[3].ravel()), (255,0,0), 5)
    return img

#venster instellen
def setwindow(windowname,width, height):
    #vensternaam instellen
    opencv.namedWindow(windowname, opencv.WINDOW_NORMAL)
    #venstergrootte instellen
    opencv.resizeWindow(windowname, width, height)

#beeld laten zien
def showimage(img, windowname):
    opencv.imshow(windowname,img)
    opencv.waitKey(1)