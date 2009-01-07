import string
import sys
import timeit
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from numpy import *

# Define some constants that we'll be using
ESCAPE = '\033'
window = 0

iteration = 0
WIDTH    = 768
HEIGHT   = 768

hotplate = arange(WIDTH*HEIGHT*3, dtype='float32').reshape(HEIGHT, WIDTH, 3)
# hotplate.fill(0.0)

for row in range(0, HEIGHT):
  for col in range(0, WIDTH):
    hotplate[row, col, 0] = 0.5
    hotplate[row, col, 1] = 0.0
    hotplate[row, col, 2] = 0.0

# Initial Conditions
hotplate[0,     0:768, 0] = 0.0
hotplate[0:768, 0,     0] = 0.0
hotplate[0:768, 767,   0] = 0.0
hotplate[767,   0:768, 0] = 1.0
hotplate[400,   0:330, 0] = 1.0
hotplate[200,   500,   0] = 1.0


def steady_state():
  global hotplate
  for row in range(1, HEIGHT - 1):
    for col in range(1, WIDTH - 1):
      avg_nearby = (hotplate[row - 1, col, 0] +
                    hotplate[row + 1, col, 0] +
                    hotplate[row, col + 1, 0] +
                    hotplate[row, col - 1, 0]) / 4
      if abs(hotplate[row, col, 0] - avg_nearby) >= 0.001:
        return False
  return True

def heat_transfer(row, col):
  global hotplate
  global WIDTH, HEIGHT
  ambient = hotplate[row - 1, col, 0] + \
            hotplate[row + 1, col, 0] + \
            hotplate[row, col + 1, 0] + \
            hotplate[row, col - 1, 0]
  return (ambient + hotplate[row, col, 0] * 4.0) / 8.0
  
def calculate():
  global hotplate, iteration
  # Temp storage for next hotplate state
  next_hotplate = hotplate.copy()
  
  ## while not steady_state():
  for row in range(1, HEIGHT - 1):
    for col in range(1, WIDTH - 1):
      next_hotplate[row, col, 0] = heat_transfer(row, col)
  
  hotplate = next_hotplate
  
  # hotplate.
  
  print hotplate
  iteration += 1
  print "Iteration: ", iteration
  DrawGLScene()

# We call this right after our OpenGL window is created.
def InitGL(width, height):
  glClearColor(0.0, 0.0, 0.0, 0.0)  # This Will Clear The Background Color To Black
  glClearDepth(1.0)                 # Enables Clearing Of The Depth Buffer
  glDepthFunc(GL_LESS)              # The Type Of Depth Test To Do
  glEnable(GL_DEPTH_TEST)           # Enables Depth Testing
  glShadeModel(GL_SMOOTH)           # Enables Smooth Color Shading
  
  glMatrixMode(GL_PROJECTION)       # Reset The Projection Matrix
  glLoadIdentity()                  # Calculate The Aspect Ratio Of The Window
  glMatrixMode(GL_MODELVIEW)
  glLoadIdentity()

  gluPerspective(45.0, float(width)/float(height), 0.1, 100.0)

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
  global hotplate
  glClearColor(0, 0, 0, 0)
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); # Clear The Screen And The Depth Buffer
  
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
  # glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_FLOAT, hotplate);
  # print hotplate
  glDrawPixelsf(GL_RGB, hotplate)
  
  # Set the state back to what it was
  glPopMatrix()
  glMatrixMode(GL_MODELVIEW)
  glPopMatrix()
  glMatrixMode(oldMatrixMode)

  if (depthWasEnabled):
    glEnable(GL_DEPTH_TEST)
    
  glutSwapBuffers()

# The function called whenever a key is pressed. Note the use of Python tuples to pass in: (key, x, y)  
def keyPressed(*args):
  # If escape is pressed, kill everything.
  if args[0] == ESCAPE:
    sys.exit()

def main():
  global WIDTH, HEIGHT
  global window
  glutInit(sys.argv)

  # Select type of Display mode:   
  #  Double buffer 
  #  RGBA color
  # Alpha components supported 
  # Depth buffer
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)

  # get a 640 x 480 window 
  glutInitWindowSize(WIDTH, HEIGHT)

  # the window starts at the upper left corner of the screen 
  glutInitWindowPosition(0, 0)

  # Get handle to window so it can be closed on exit
  window = glutCreateWindow("CS 484 : Project 1 / Hotplate")

  # Register the drawing function with glut, BUT in Python land, at least using PyOpenGL, we need to
  # set the function pointer and invoke a function to actually register the callback, otherwise it
  # would be very much like the C version of the code.  
  glutDisplayFunc(DrawGLScene)

  # Uncomment this line to get full screen.
  # glutFullScreen()

  # When we are doing nothing, redraw the scene.
  glutIdleFunc(calculate)
  # glutIdleFunc(DrawGLScene)

  # Register the function called when our window is resized.
  glutReshapeFunc(ReSizeGLScene)

  # Register the function called when the keyboard is pressed.  
  glutKeyboardFunc(keyPressed)

  # Initialize our window. 
  InitGL(WIDTH, HEIGHT)

  # Start Event Processing Engine 
  glutMainLoop()

# Print message to console, and kick off the main to get it rolling.
print "Hit ESC key to quit."
main()
