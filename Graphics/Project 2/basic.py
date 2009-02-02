import string
import sys
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from DuaneGL import *

def init():
  glClearColor(1.0, 1.0, 1.0, 0.0)
  glMatrixMode(GL_PROJECTION)
  # gluOrtho2D(0.0,640.0, 0.0,480.0)
  # glClearDepth(1.0)
  glDisable(GL_DEPTH_TEST)
  glOrtho(0,640, 0,480, -1,1);
  # glRasterPos2i(-50, -50)

def draw():
  glClear(GL_COLOR_BUFFER_BIT)
  
  drawRaster()
  # drawDemo()

def drawRaster():
  djClearColor(0.1, 0.1, 0.1)
  djClear(GL_COLOR_BUFFER_BIT)
  
  djColor3f(1.0, 0, 0)
  djDrawPoint(320, 240)
  
  oldMatrixMode = glGetIntegerv(GL_MATRIX_MODE)
  
  glMatrixMode(GL_MODELVIEW)
  glPushMatrix()
  glLoadIdentity()
  
  glMatrixMode(GL_PROJECTION)
  glPushMatrix()
  glLoadIdentity()
  
  glRasterPos2f(-1,-1)
  glDrawPixels(640, 480, GL_RGB, GL_FLOAT, raster);
  
  glFlush()
  
  glPopMatrix()
  glMatrixMode(GL_MODELVIEW)
  
  glPopMatrix()
  glMatrixMode(oldMatrixMode)
  
  # glutSwapBuffers()

def drawDemo():
  glBegin(GL_LINES)
  for i in range(8):
    print("i is ", i)
    glColor3f(1.0, 0.0, 0.0)
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
  

glutInit(sys.argv)
glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)
glutInitWindowPosition(50, 100)
glutInitWindowSize(640, 480)
glutCreateWindow("A Basic OpenGL Program in Python")

init()
glutDisplayFunc(draw)
glutMainLoop()