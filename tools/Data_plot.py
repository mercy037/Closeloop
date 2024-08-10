
# #  画动态曲线

import matplotlib.pyplot as plt
from random import random


def do_something(res=0):
    for p in range(100):
        res += p


fig, ax = plt.subplots()
x = []
y = []
res = 0
for i in range(50):
    x.append(i)
    y.append(50 * random())
    ax.cla()  # clear plot
    ax.plot(x, y, 'r', lw=1)  # draw line chart
    # ax.bar(x, height=y, width=0.3) # draw bar chart
    do_something()
    plt.pause(0.1)