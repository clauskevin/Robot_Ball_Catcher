# Robot Ball Catcher
### Bachelorproject Elektromechanica 2021: EM05 Robot Ball Catcher - KU Leuven Campus Brugge

This is a project about developing an algorithm for a robot to catch a ball with the help of a depth camera.
The camera I used for this project is a Intel RealSense D415.
The code discription comments are written in dutch since this is a university project in my own language.

## Installation
- Install Python 3.6
- Download this repository as .zip
- open this repository as a project in a Python IDE (I use Pycharm)
- pip install libraries: pyrealsense2, cv2, numpy, matplotlib and scipy
- Run the scripts

## Structure
### src
In this folder, you can find the main class (Robot_Ball_Catcher_Functions.py).
This file contains functions that are used in the different scripts.
### data
This folder contains .npy files. These are Numpy arrays, used to save and share data between scripts.
The data folder also contains the chessboard pattern, used for callibration.
### script
This folder contains all the project scripts to run.

#### 1. Camera viewer
Display color image and depth colormap
#### 2. HSV calibration tool
Search for upper and lower HSV color
#### 3. UVZ ball detection
2D ball detection + depth
#### 4. Intrinsic calibration tool
Intrinsic calibration
#### 5. Extrinsic calibration tool
Extrinsic calibration
#### 6. XYZ ball detection
3D ball detection
#### 7. XYZ realtime plot
realtime 3D coordinate graph plotting
#### 8. UVZ recorder
Record 2D ball information + depth
#### 9. XYZ asynchronous plot
3D coordinate graph plotting of recorded ball information
#### 10. Regression simulation
Simulating the course of the regression during the ball throw.

## Extra
Be aware that this project uses a lot of manual configured variables. Also the .npy array files will be overwrited when running the scripts.

