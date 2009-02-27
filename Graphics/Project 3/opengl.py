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
  clear(GL_COLOR_BUFFER_BIT)
  disable(GL_POINT_SMOOTH)
  pointSize(1)
  lineWidth(1)
  
def vx(value):
  return (float(value) - 320) / 640

def vy(value):
  return (float(value) - 240) / 480

def scene_a():
  scene_clear()
  radius = 90.0
  
  viewport(0, 0, 480, 480)
  
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
    vertex2i(int(320 + radius*math.sin(r)), int(240 + radius*math.cos(r)))
    color3f(0.5, 0.5, 1.0)
    vertex2i(int(320 + radius*math.sin(r+step)), int(240 + radius*math.cos(r+step)))
  end()

def scene_c():
  scene_clear()
  radius = 10.0
  pointSize(10)
  lineWidth(6)
  
  # Draw square points
  disable(GL_POINT_SMOOTH)
  begin(GL_POINTS)
  rgb = [[1.0, 1.0, 0.0], [1.0, 0.5, 0.0], [0.5, 0.0, 1.0]]
  for i in range(3):
    apply(color3f, rgb[i])
    vertex2i(260 + i*40, 240)
  end()
  
  # Now draw circular points
  enable(GL_POINT_SMOOTH)
  begin(GL_POINTS)
  for i in range(3):
    apply(color3f, rgb[i])
    vertex2i(260 + i*40, 200)
  end()
  
  # Draw a thick line
  begin(GL_LINES)
  apply(color3f, rgb[0])
  vertex2i(260, 150)
  apply(color3f, rgb[1])
  vertex2i(380, 160)
  end()

def scene_d():
  scene_clear()
  begin(GL_LINE_STRIP)
  color3f(1.0, 0.0, 0.0)
  vertex2i(320, 140)
  color3f(1.0, 1.0, 0.0)
  vertex2i(380, 100)
  color3f(1.0, 0.0, 0.0)
  vertex2i(330, 60)
  color3f(1.0, 1.0, 0.0)
  vertex2i(200, 80)
  end()

  begin(GL_LINE_LOOP)
  color3f(1.0, 0.0, 0.0)
  vertex2i(320, 240)
  color3f(1.0, 1.0, 0.0)
  vertex2i(380, 200)
  color3f(1.0, 0.0, 0.0)
  vertex2i(330, 160)
  color3f(1.0, 1.0, 0.0)
  vertex2i(200, 180)
  end()
  
  begin(GL_TRIANGLE_STRIP)
  color3f(0.0, 0.5, 0.0)
  vertex2i(100, 380)
  color3f(1.0, 0.5, 0.0)
  vertex2i(120, 370)
  color3f(0.0, 0.5, 1.0)
  vertex2i(140, 390)

  color3f(0.0, 0.5, 0.0)
  vertex2i(140, 390)
  color3f(1.0, 0.5, 0.0)
  vertex2i(120, 370)
  color3f(0.0, 0.5, 1.0)
  vertex2i(180, 380)

  color3f(1.0, 0.0, 0.0)
  vertex2i(180, 380)
  color3f(0.0, 0.0, 1.0)
  vertex2i(120, 370)
  color3f(1.0, 0.0, 1.0)
  vertex2i(200, 350)
  end()
  
  radius = 50.0
  step = math.pi/8
  begin(GL_TRIANGLE_FAN)
  color3f(1.0, 0.0, 0.0)
  vertex2i(400, 400)
  color3f(0.0, 0.0, 1.0)
  vertex2i(450, 400)
  n = 0
  for r in frange(math.pi + step, 2*math.pi - step, step):
    if (n % 2 == 0):
      color3f(1.0, 0.5, 0.0)
    else:
      color3f(1.0, 1.0, 1.0)
    vertex2i(int(425 + radius*math.sin(r)), int(400 - radius*math.cos(r)))
    n += 1
  end()
  
  begin(GL_QUADS)
  color3f(0.0, 0.5, 0.0)
  vertex2i(450, 380)
  color3f(1.0, 0.5, 0.0)
  vertex2i(440, 360)
  color3f(0.0, 0.5, 1.0)
  vertex2i(500, 365)
  color3f(1.0, 1.0, 1.0)
  vertex2i(510, 390)

  color3f(0.0, 0.5, 0.0)
  vertex2i(450, 330)
  color3f(1.0, 0.5, 0.0)
  vertex2i(440, 310)
  color3f(0.0, 0.5, 1.0)
  vertex2i(500, 315)
  color3f(1.0, 1.0, 1.0)
  vertex2i(510, 340)
  end()

  begin(GL_QUAD_STRIP)
  color3f(0.0, 0.5, 0.0)
  vertex2i(450, 180)
  color3f(1.0, 0.5, 0.0)
  vertex2i(440, 160)
  color3f(0.0, 0.5, 1.0)
  vertex2i(500, 165)
  color3f(1.0, 1.0, 1.0)
  vertex2i(510, 190)

  color3f(0.0, 0.5, 0.0)
  vertex2i(550, 180)
  color3f(1.0, 0.5, 0.0)
  vertex2i(560, 160)
  
  color3f(0.0, 0.5, 0.0)
  vertex2i(600, 185)
  color3f(1.0, 0.5, 0.0)
  vertex2i(610, 170)
  end()
  
  
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
  
  DrawGLScene()

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