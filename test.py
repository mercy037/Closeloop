import pandas as pd
import numpy as np
import heapq
distance = [1,3,7,9,9]
min1,min2,min3,min4,min5 = heapq.nsmallest(5, distance)

print(min1,min2,min3,min4,min5)
