from numpy import *

m = array([[1, 2, 3, 4], [0, 5, 6, 7], [0, 0, 8, 9], [0, 0, 0, 0]])
n = array([[1, 0, 1, 0], [0, 2, 0, 2], [0, 0, 3, 0], [0, 0, 0, 4]])
i = matrix("0 1 1; 1 0 1; 1 1 0")

print "Initial array"
print m
print

print "Flatten"
print m.transpose().flatten()
print

print "Add one"
print (m + 1)
print

print "Add vector [1, 0, 1, 0]"
print m + array([1, 0, 1, 0])
print

print "Subtract vector [1, 0, 1, 0]"
print m - array([1, 0, 1, 0])
print

print "Multiply m * n"
print matrix(m) * matrix(n)
print

print "Multiply m * [2, 3, 5, 7]"
print dot(m, [2, 3, 5, 7])
print

print "Inner product of [1, 2, 3] and [1, 1, 1]"
print dot([1, 2, 3], [1, 1, 1])
print

print "Matrix inversion"
print i
print i.I
print