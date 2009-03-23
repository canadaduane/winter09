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
x = 0.0
y = 0.01
z = -2.5
w = 0.1
d = 1.0


# Helper Function for floating-point ranges
def frange(fr, to, step):
    while fr < to:
       yield fr
       fr += step

def scene_clear():
  clearColor(0.0, 0.0, 0.0)
  clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
  # disable(GL_POINT_SMOOTH)
  # pointSize(1)
  # lineWidth(1)
  matrixMode(GL_MODELVIEW)
  loadIdentity()
  color3f(0.5, 0.5, 1.0)
  
def vx(value):
  return (float(value) - 320) / 640

def vy(value):
  return (float(value) - 240) / 480

def scene_a():
  global x, y, z, w, d
  scene_clear()  
  
  enable(GL_CULL_FACE)
  frontFace(GL_CCW)
  cullFace(GL_BACK)
  
  translate(x, y, z)
  begin(GL_QUADS)
  # normal3f(1.0, 1.0, 1.0)
  color3f(1.0, 0.0, 0.0)
  vertex3f(-w/2,  0,  -d/2)
  vertex3f( w/2,  0,  -d/2)
  vertex3f( w/2,  0.1,   d/2)
  vertex3f(-w/2,  0.1,   d/2)
  end()
  
  begin(GL_TRIANGLES)
  color3f(1.0, 0.0, 0.0)
  vertex3f(-0.2, -0.2, -1.0)
  color3f(1.0, 1.0, 0.0)
  vertex3f(-0.8, -0.8, -1.0)
  color3f(0.0, 0.0, 1.0)
  vertex3f(-0.15, -0.5, -1.0)

  color3f(1.0, 0.0, 0.0)
  vertex3f(0.2, -0.2, -1.0)
  color3f(1.0, 1.0, 0.0)
  vertex3f(0.8, -0.8, -1.0)
  color3f(0.0, 0.0, 1.0)
  vertex3f(0.15, -0.5, -1.0)
  end()

def scene_b():
  scene_clear()
  radius = 90.0
  
  # viewport(0, 0, 640, 480)
  
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

def scene_c():
  scene_clear()
  
  enable(GL_NORMALIZE)
  enable(GL_LIGHTING)
  enable(GL_LIGHT0)
  light(GL_LIGHT0, GL_DIFFUSE, [0.5, 0.0, 0.0, 1.0])
  light(GL_LIGHT0, GL_AMBIENT, [0.1, 0.1, 0.1, 1.0])
  light(GL_LIGHT0, GL_POSITION, [0.5, 0.5, -3.0, 0.0])
  
  dp = math.pi/8 # 16 picked arbitrarily; try other numbers too
  translate(0, 0, -5.0)
  # begin(GL_TRIANGLES)
  # normal3f(0.0, 0.0, -1.0)
  # vertex3f(0.0, 0.2, 0.0)
  # vertex3f(-0.2, -0.1, 0.0)
  # vertex3f(0.2, 0.0, 0.0)
  # end()
  
  begin(GL_QUADS)
  for theta in frange(0, 2*math.pi, dp):
    for phi in frange(0, math.pi, dp):
      # color3f(1.0, 0.0, 0.0)
      normal3f(math.cos(theta)   *math.sin(phi)*2,    math.cos(phi)*2,    math.sin(theta)   *math.sin(phi)*2)
      vertex3f(math.cos(theta)   *math.sin(phi),    math.cos(phi),    math.sin(theta)   *math.sin(phi))
      # color3f(0.0, 1.0, 0.0)
      normal3f(math.cos(theta+dp)*math.sin(phi)*2,    math.cos(phi)*2,    math.sin(theta+dp)*math.sin(phi)*2)
      vertex3f(math.cos(theta+dp)*math.sin(phi),    math.cos(phi),    math.sin(theta+dp)*math.sin(phi))
      # color3f(1.0, 1.0, 0.0)
      normal3f(math.cos(theta+dp)*math.sin(phi+dp)*2, math.cos(phi+dp)*2, math.sin(theta+dp)*math.sin(phi+dp)*2)
      vertex3f(math.cos(theta+dp)*math.sin(phi+dp), math.cos(phi+dp), math.sin(theta+dp)*math.sin(phi+dp))
      # color3f(0.0, 0.0, 1.0)
      normal3f(math.cos(theta)   *math.sin(phi+dp)*2, math.cos(phi+dp)*2, math.sin(theta)   *math.sin(phi+dp)*2)
      vertex3f(math.cos(theta)   *math.sin(phi+dp), math.cos(phi+dp), math.sin(theta)   *math.sin(phi+dp))
  end()
  
  # glEnable(GL_LIGHTING)
  # glEnable(GL_LIGHT0)
  # glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.5, 0.0, 0.0, 1.0])
  # glLightfv(GL_LIGHT0, GL_AMBIENT, [0.1, 0.1, 0.1, 1.0])
  # glLightfv(GL_LIGHT0, GL_POSITION, [0.5, 0.5, -3.0, 0.0])
  # 
  # dp = math.pi/4 # 16 picked arbitrarily; try other numbers too
  # glTranslatef(0, 0, -5.0)
  # glBegin(GL_QUADS)
  # for theta in frange(0, 2*math.pi, dp):
  #   for phi in frange(0, math.pi, dp):
  #     glNormal3f(math.cos(theta)   *math.sin(phi),    math.cos(phi),    math.sin(theta)   *math.sin(phi))
  #     glVertex3f(math.cos(theta)   *math.sin(phi),    math.cos(phi),    math.sin(theta)   *math.sin(phi))
  #     glNormal3f(math.cos(theta+dp)*math.sin(phi),    math.cos(phi),    math.sin(theta+dp)*math.sin(phi))
  #     glVertex3f(math.cos(theta+dp)*math.sin(phi),    math.cos(phi),    math.sin(theta+dp)*math.sin(phi))
  #     glNormal3f(math.cos(theta+dp)*math.sin(phi+dp), math.cos(phi+dp), math.sin(theta+dp)*math.sin(phi+dp))
  #     glVertex3f(math.cos(theta+dp)*math.sin(phi+dp), math.cos(phi+dp), math.sin(theta+dp)*math.sin(phi+dp))
  #     glNormal3f(math.cos(theta)   *math.sin(phi+dp), math.cos(phi+dp), math.sin(theta)   *math.sin(phi+dp))
  #     glVertex3f(math.cos(theta)   *math.sin(phi+dp), math.cos(phi+dp), math.sin(theta)   *math.sin(phi+dp))
  # glEnd()

def scene_d():
  scene_clear()

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
  
  glViewport(0, 0, width, height)
  
  # Reset The Current Viewport And Perspective Transformation
  # glViewport(0, 0, width, height)
  glMatrixMode(GL_PROJECTION)
  glLoadIdentity()
  # print projectionMatrix(True)
  # glFrustum(-1.0,1.0, -1.0*float(height)/float(width),1.0*float(height)/float(width), 0.1, 100)
  # frustum(-1.0,1.0, -1.0*float(height)/float(width),1.0*float(height)/float(width),  0.1,100)
  frustum(-0.1,0.1, -0.1*480/640,0.1*480/640,  0.1,10)
  # print projectionMatrix(True)
  # gluPerspective(45, float(width)/float(height), 0.1, 100)
  # perspective(45, float(width)/float(height), 0.1, 100)
  
  glMatrixMode(GL_MODELVIEW)
  glLoadIdentity()

def DrawGLScene():
  global sceneChoice
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); # Clear The Screen And The Depth Buffer
  glClearColor(0.0, 0.0, 0.0, 0.0)
  
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
  global x, y, z, w, d
  
  def set_draw_mode(m):
    global drawMode
    drawMode = m
  
  def set_scene_choice(m):
    global sceneChoice
    print "set_scene_choice"
    sceneChoice = m
  
  def right(amount):
    global x
    x += amount
  
  def up(amount):
    global y
    y += amount

  def into(amount):
    global z
    z += amount
  
  try:
    # If escape is pressed, kill everything.
    {
      ESCAPE: sys.exit,
      '1': lambda: set_draw_mode(1),
      '2': lambda: set_draw_mode(2),
    
      'a': lambda: set_scene_choice(0),
      's': lambda: set_scene_choice(1),
      'd': lambda: set_scene_choice(2),
      'f': lambda: set_scene_choice(3),
      
      'i': lambda: up(0.02),
      'k': lambda: up(-0.02),
      'j': lambda: right(-0.02),
      'l': lambda: right(0.02),
      'u': lambda: into(0.5),
      'h': lambda: into(-0.5)
      
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