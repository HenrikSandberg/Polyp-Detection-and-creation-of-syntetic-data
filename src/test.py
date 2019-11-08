import numpy as np
import random

grid = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
random.shuffle(grid)
print(grid)

split = int(len(grid)*0.85)
x = grid[0: split]
y = grid[split:]
print(split)
print(x)
print(y)