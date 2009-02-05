import string
import sys
import time
import math
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from numpy import *

import duanegl
from duanegl import *

# Define some constants that we'll be using
ESCAPE = '\033'
window = 0
drawMode = 1

clearColor(0.2, 0.2, 0.2)
clear(GL_COLOR_BUFFER_BIT)

def frange(fr, to, step):
    while fr < to:
       yield fr
       fr += step

# Test lines
red = [1.0, 0.0, 0.0, 1.0]
white = [1.0, 1.0, 1.0, 1.0]
blue = [0.0, 0.0, 1.0, 1.0]
green = [0.0, 1.0, 0.0, 1.0]

p1 = [320.2, 240.0]
p2 = [200.9, 284.4]
drawPerfectYLine(p1, p2, red, white)

apply(color4f, green)
apply(drawPoint2f, p1)
apply(drawPoint2f, p2)

# drawPerfectLine([340.9, 284.4], [300.2, 275.8], white, white)
# drawPerfectLine([300.2, 275.8], [320.2, 240.0], white, white)
# begin(GL_TRIANGLES)
# for r in frange(0, 2*math.pi, math.pi/8):
#   color3f(1.0, 0.5, 0.0)
#   vertex2i(320, 240)
#   color3f(0.5, 0.5, 1.0)
#   vertex2i(int(320 + 40*math.sin(r)), int(240 + 40*math.cos(r)))
# end()

# Test triangles
# begin(GL_TRIANGLES)
# color3f(1,0,0)
# vertex2i(300,300)
# color3f(0,1,0)
# vertex2i(500,300)
# color3f(0,0,1)
# vertex2i(400,350)
# 
# color3f(1,0,0)
# vertex2i(20,40)
# color3f(0,1,0)
# vertex2i(30,30)
# color3f(0,0,1)
# vertex2i(120,50)
# end()


# We call this right after our OpenGL window is created.
def InitGL(width, height):
  glClearColor(0.0, 0.0, 0.0, 0.0)  # This Will Clear The Background Color To Black
  glClearDepth(1.0)                 # Enables Clearing Of The Depth Buffer
  glDepthFunc(GL_LESS)              # The Type Of Depth Test To Do
  glEnable(GL_DEPTH_TEST)           # Enables Depth Testing
  glShadeModel(GL_SMOOTH)           # Enables Smooth Color Shading
  
def ReSizeGLScene(width, height):
  # Prevent A Divide By Zero If The Window Is Too Small 
  if height == 0:
    height = 1

  # Reset The Current Viewport And Perspective Transformation
  glViewport(0, 0, width, height)
  glMatrixMode(GL_PROJECTION)
  glLoadIdentity()
  glMatrixMode(GL_MODELVIEW)
  glLoadIdentity()
  glOrtho(0,width, 0,height, -1,1);

def DrawGLScene():
  glClearColor(0.1, 0.1, 0.1, 1)
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); # Clear The Screen And The Depth Buffer
  if drawMode == 1:
    print "Inside drawMode 1"
    
    # glBegin(GL_LINES)
    # for i in range(8):
    #   glColor3f(1.0,0.0,0.0)
    #   glVertex2i(200,200)
    #   glVertex2i(200 + 10*i, 280)
    #   glColor3f(0.0,1.0,0.0)
    #   glVertex2i(200,200)
    #   glColor3f(0.0,1.0,1.0)
    #   glVertex2i(200 - 10*i, 280)
    #   glVertex2i(200,200)
    #   glVertex2i(280, 200 + 10*i)
    #   glVertex2i(200,200)
    #   glVertex2i(280, 200 - 10*i)
    # glEnd()
    
  elif drawMode == 2:
    depthWasEnabled = glIsEnabled(GL_DEPTH_TEST)
    glDisable(GL_DEPTH_TEST)
    oldMatrixMode = glGetIntegerv(GL_MATRIX_MODE)
    
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    
    # Draw the raster image
    glRasterPos2f(-1,-1)
    glDrawPixels(640, 480, GL_RGB, GL_FLOAT, duanegl.raster);
    
    # Set the state back to what it was
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    
    glPopMatrix()
    glMatrixMode(oldMatrixMode)
    # 
    if (depthWasEnabled):
      glEnable(GL_DEPTH_TEST)

  glFlush()
    
  glutSwapBuffers()

# The function called whenever a key is pressed. Note the use of Python tuples to pass in: (key, x, y)  
def keyPressed(*args):
  global drawMode
  # If escape is pressed, kill everything.
  if args[0] == ESCAPE:
    sys.exit()
  if args[0] == '1':
    drawMode = 1
  if args[0] == '2':
    drawMode = 2
  print("drawMode: ", drawMode)

def main():
  global window
  glutInit(sys.argv)

  # Select type of Display mode:   
  #  Double buffer 
  #  RGBA color
  # Alpha components supported 
  # Depth buffer
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)

  # get a 640 x 480 window 
  glutInitWindowSize(640, 480)

  # the window starts at the upper left corner of the screen 
  glutInitWindowPosition(0, 0)

  # Get handle to window so it can be closed on exit
  window = glutCreateWindow("CS 455 : Project 1")

  # Register the drawing function with glut, BUT in Python land, at least using PyOpenGL, we need to
  # set the function pointer and invoke a function to actually register the callback, otherwise it
  # would be very much like the C version of the code.  
  glutDisplayFunc(DrawGLScene)

  # Uncomment this line to get full screen.
  # glutFullScreen()
  
  def sleep():
    time.sleep(1)
    DrawGLScene
  # When we are doing nothing, redraw the scene.
  glutIdleFunc(sleep)

  # Register the function called when our window is resized.
  glutReshapeFunc(ReSizeGLScene)

  # Register the function called when the keyboard is pressed.  
  glutKeyboardFunc(keyPressed)

  # Initialize our window. 
  InitGL(640, 480)

  # Start Event Processing Engine 
  glutMainLoop()

# Print message to console, and kick off the main to get it rolling.
print "Hit ESC key to quit."
main()