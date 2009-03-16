import string
import sys
import time
import math
import random
from numpy import *
from functional import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import duanegl
from duanegl import *

# Define some constants that we'll be using
ESCAPE = '\033'
window = 0
drawMode = 2
sceneChoice = 0

# Helper Function for floating-point ranges
def frange(fr, to, step):
    while fr < to:
       yield fr
       fr += step

def scene_clear():
  clearColor(0.0, 0.0, 0.0)
  clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
  disable(GL_POINT_SMOOTH)
  pointSize(1)
  lineWidth(1)
  matrixMode(GL_MODELVIEW)
  loadIdentity()
  
def vx(value):
  return (float(value) - 320) / 640

def vy(value):
  return (float(value) - 240) / 480

def scene_a():
  scene_clear()
  matrixMode(GL_PROJECTION)
  loadIdentity()
  # frustum(-0.1,0.1, -0.1*480/640,0.1*480/640,  0.1,10)
  perspective(90, double(640)/480, 0.1, 10)
  matrixMode(GL_MODELVIEW)
  loadIdentity()
  
  begin(GL_TRIANGLES)
  color3f(1.0, 0.0, 0.0)
  vertex3f(0.0, 0.0, -9.0)
  vertex3f(-1.0, -0.5, -5.0)
  vertex3f(1.0, -0.5, -5.0)
  end()

def scene_e():
  scene_clear()
  radius = 90.0
  
  viewport(0, 0, 640, 480)
  
  begin(GL_POINTS)
  color3f(1.0, 1.0, 1.0)
  vertex2f(vx(50), vy(50))
  end()
  
  begin(GL_TRIANGLES)
  for r in frange(0, 2*math.pi, math.pi/4):
    color3f(1.0, 0.0, 0.0)
    x, y = vx(320), vy(240)
    vertex2f(x, y)

    color3f(0.0, 1.0, 0.0)
    x, y = vx(320 + radius*math.sin(r)), vy(240 + radius*math.cos(r))
    vertex2f(x, y)

    color3f(0.0, 0.0, 1.0)
    x, y = vx(320 + radius*math.sin(r+math.pi/8)), vy(240 + radius*math.cos(r+math.pi/8))
    vertex2f(x, y)
  end()
  

def scene_b():
  scene_clear()
  radius = 50.0
  step = math.pi/8
  begin(GL_LINES)
  for r in frange(0, 2*math.pi, step):
    color3f(1.0, 0.5, 0.0)
    vertex2f(vx(320 + radius*math.sin(r)), vy(240 + radius*math.cos(r)))
    color3f(0.5, 0.5, 1.0)
    vertex2f(vx(320 + radius*math.sin(r+step)), vy(240 + radius*math.cos(r+step)))
  end()

def scene_c():
  scene_clear()
  
  matrixMode(GL_MODELVIEW)
  loadIdentity()
  rotate(15, 0, 0, 1.0)
  
  begin(GL_TRIANGLES)
  color3f(1.0, 0.5, 0.0)
  vertex3f(-0.5, 0.2, 0.5)
  color3f(1.0, 0.8, 0.0)
  vertex3f(0.0, -0.5, 0.0)
  color3f(1.0, 0.2, 0.0)
  vertex3f(0.5, 0.2, -0.5)
  
  color3f(0.0, 0.2, 1.0)
  vertex3f(-0.5, -0.2, -0.5)
  color3f(0.0, 0.4, 1.0)
  vertex3f(0.0, 0.5, 0.0)
  color3f(0.0, 0.1, 1.0)
  vertex3f(0.5, -0.2, 0.5)
  end()

def tree(depth):
  r2 = 1.0/sqrt(2);
  mdown = matrix(array([[0,-r2,0,0], [r2,0,0,0], [0,0,1,0], [0,-r2,0,1]], dtype='float32')).transpose()
  mup = matrix(array([[0,r2,0,0], [-r2,0,0,0], [0,0,1,0], [0,r2,0,1]], dtype='float32')).transpose()
  
  if (depth <= 0):
    return
  
  begin(GL_LINES)
  vertex2f(0,-r2)
  vertex2f(0, r2)
  end()
  
  pushMatrix()
  multMatrix(mdown)
  tree(depth-1)
  popMatrix()
  pushMatrix()
  multMatrix(mup)
  tree(depth-1)
  popMatrix()

def scene_d():
  scene_clear()
  viewport(0, 0, 640, 480)
  
  # loadMatrix([1, 0, 0, 0,     0, -1, 0, 0,   0, 0, 1, 0,   0, 0, 0, 1])
  # multMatrix([0.5, 0, 0, 0,   0, 1, 0, 0,    0, 0, 1, 0,   0, 0, 0, 1])
  # print current_matrix
  # print modelview_matrix_stack
  
  ortho(-1, 2, -1, 2, -1, 2)
  tree(8)
  
  # rotate(45, 0.5, 0, 1.0)
  # translate(0.2, 0.5, 0)
  # scale(0.5, 0.5, 1.0)
  # fixed_scale(0.6, 0.6, 0.6,  0.0, 0.0, 0.0)
  # shear(0.5, 0.5, 0.0, 0.0, 0.0, 0.0)
    
  # begin(GL_TRIANGLES)
  # color3f(1.0, 0.5, 0.0)
  # vertex3f(-0.5, 0.2, 0.5)
  # color3f(1.0, 0.8, 0.0)
  # vertex3f(0.0, -0.5, 1.3)
  # color3f(1.0, 0.2, 0.0)
  # vertex3f(0.5, 0.2, -0.5)
  # end()

  # begin(GL_TRIANGLES)
  # color3f(1.0, 0.5, 0.0)
  # vertex3f(0.0, 0.0, 0.0)
  # color3f(1.0, 0.8, 0.0)
  # vertex3f(0.0, 2.0, 0.0)
  # color3f(2.0, 2.0, 0.0)
  # vertex3f(0.5, 0.2, -0.5)
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
  # glViewport(0, 0, width, height)
  glMatrixMode(GL_PROJECTION)
  glLoadIdentity()
  glMatrixMode(GL_MODELVIEW)
  glLoadIdentity()
  # glOrtho(0,width, 0,height, -1,1);
  
  # DrawGLScene()

def DrawGLScene():
  global sceneChoice
  glClearColor(0.1, 0.1, 0.1, 1)
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); # Clear The Screen And The Depth Buffer
  
  # Draw the chosen scene
  [scene_a, scene_b, scene_c, scene_d][sceneChoice]()
  
  if drawMode == 2:
    
    clippingWasOn = [glIsEnabled(GL_CLIP_PLANE0 + i) for i in range(6)]
    [glDisable(GL_CLIP_PLANE0+i) for i in range(6)]
    
    depthWasEnabled = glIsEnabled(GL_DEPTH_TEST)
    glDisable(GL_DEPTH_TEST)

    oldViewport = glGetIntegerv(GL_VIEWPORT)
    glViewport(0,0,640,480);
    
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
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()

    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()

    glMatrixMode(oldMatrixMode)
    
    glViewport(oldViewport[0], oldViewport[1], oldViewport[2], oldViewport[3])
    
    if (depthWasEnabled):
      glEnable(GL_DEPTH_TEST)
    
    for i in range(6):
      if clippingWasOn[i]:
        glEnable(GL_CLIP_PLANE0 + i)
    
  glFlush()
    
  glutSwapBuffers()

# The function called whenever a key is pressed. Note the use of Python tuples to pass in: (key, x, y)  
def keyPressed(*args):
  global drawMode, sceneChoice
  
  def set_draw_mode(m):
    global drawMode
    drawMode = m
  
  def set_scene_choice(m):
    global sceneChoice
    print "set_scene_choice"
    sceneChoice = m
  
  try:
    # If escape is pressed, kill everything.
    {
      ESCAPE: sys.exit,
      '1': lambda: set_draw_mode(1),
      '2': lambda: set_draw_mode(2),
    
      'a': lambda: set_scene_choice(0),
      's': lambda: set_scene_choice(1),
      'd': lambda: set_scene_choice(2),
      'f': lambda: set_scene_choice(3)
    }[args[0]]()
  except KeyError:
    # that's ok
    print "Unknown key:", args[0]
  
  print("drawMode:", drawMode, "sceneChoice:", sceneChoice)
  DrawGLScene()

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
    time.sleep(0.2)
    
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