import sys
import timeit
from numpy import *

MAX_TEMP = 100
WIDTH    = 768
HEIGHT   = 768

hotplate = arange(WIDTH*HEIGHT).reshape(HEIGHT, WIDTH)
hotplate.fill(50.0)

# Initial Conditions
hotplate[0,     0:768] = 0.0
hotplate[0:768, 0]     = 0.0
hotplate[0:768, 767]   = 0.0
hotplate[767,   0:768] = 100.0
hotplate[400,   0:330] = 100.0
hotplate[200,   500]   = 100.0

def steady_state():
  global hotplate
  for row in range(1, HEIGHT - 1):
    for col in range(1, WIDTH - 1):
      avg_nearby = (hotplate[row - 1, col] +
                    hotplate[row + 1, col] +
                    hotplate[row, col + 1] +
                    hotplate[row, col - 1]) / 4
      if abs(hotplate[row, col] - avg_nearby) >= 0.1:
        return False
  return True

def heat_transfer(row, col):
  global hotplate
  global WIDTH, HEIGHT
  ambient = hotplate[row - 1, col] + hotplate[row + 1, col] + hotplate[row, col + 1] + hotplate[row, col - 1]
  return (ambient + hotplate[row, col] * 4) / 8
  
def run():
  global hotplate
  # Temp storage for next hotplate state
  next_hotplate = hotplate.copy()
  
  n = 0
  while not steady_state():
    print hotplate
    for row in range(1, HEIGHT - 1):
      for col in range(1, WIDTH - 1):
        next_hotplate[row, col] = heat_transfer(row, col)
    n += 1
    print "Iter: ", n
    hotplate = next_hotplate
  
  print "Final solution:"
  print hotplate

run()