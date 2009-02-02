import string
import sys
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from numpy import *

import DuaneGL
from DuaneGL import *

# Define some constants that we'll be using
ESCAPE = '\033'
window = 0
drawMode = 1

clearColor(0.2, 0.2, 0.2)
clear(GL_COLOR_BUFFER_BIT)

color3f(1.0, 0.0, 0.0)
begin(GL_POINTS)
vertex2i(320, 240)
vertex2i(322, 240)

# We call this right after our OpenGL window is created.
def InitGL(width, height):
  glClearColor(0.0, 0.0, 0.0, 0.0)  # This Will Clear The Background Color To Black
  glClearDepth(1.0)                 # Enables Clearing Of The Depth Buffer
  glDepthFunc(GL_LESS)              # The Type Of Depth Test To Do
  glEnable(GL_DEPTH_TEST)           # Enables Depth Testing
  glShadeModel(GL_SMOOTH)           # Enables Smooth Color Shading
  
  glMatrixMode(GL_PROJECTION)       # Reset The Projection Matrix
  # glLoadIdentity()                  # Calculate The Aspect Ratio Of The Window
  # glMatrixMode(GL_MODELVIEW)
  # glLoadIdentity()
  glOrtho(0,640, 0,480, -1,1);

  # gluPerspective(45.0, float(width)/float(height), 0.1, 100.0)

def ReSizeGLScene(width, height):
  # Prevent A Divide By Zero If The Window Is Too Small 
  if height == 0:
    height = 1

  # Reset The Current Viewport And Perspective Transformation
  glViewport(0, 0, width, height)
  glMatrixMode(GL_PROJECTION)
  glLoadIdentity()
  gluPerspective(45.0, float(width) / float(height), 0.1, 100.0)
  glMatrixMode(GL_MODELVIEW)

def DrawGLScene():
  glClearColor(0.1, 0.1, 0.1, 1)
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); # Clear The Screen And The Depth Buffer
  if drawMode == 1:
    oldMatrixMode = glGetIntegerv(GL_MATRIX_MODE)
    
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    
    glBegin(GL_LINES)
    for i in range(8):
      glColor3f(1,0,0)
      glVertex2i(200,200)
      glVertex2i(200 + 10*i, 280)
      glColor3f(0,1,0)
      glVertex2i(200,200)
      glColor3f(0,1,1)
      glVertex2i(200 - 10*i, 280)
      glVertex2i(200,200)
      glVertex2i(280, 200 + 10*i)
      glVertex2i(200,200)
      glVertex2i(280, 200 - 10*i)
    glEnd()

    glFlush()
    
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    
    glPopMatrix()
    glMatrixMode(oldMatrixMode)
    
    
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
    glDrawPixels(640, 480, GL_RGB, GL_FLOAT, DuaneGL.raster);
    
    # Set the state back to what it was
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    
    glPopMatrix()
    glMatrixMode(oldMatrixMode)
    # 
    if (depthWasEnabled):
      glEnable(GL_DEPTH_TEST)
    
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

  # When we are doing nothing, redraw the scene.
  glutIdleFunc(DrawGLScene)

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