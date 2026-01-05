#
# Intro to matplotlib
#
# Author: Nick van Remortel
#
# Last updated: 29 March 2021
#
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.style as mplstyle

# optional: set style to fast ane efficient for basic plotting
# mplstyle.use('fast')
matplotlib.use('TkAgg')
mplstyle.use(['dark_background', 'ggplot', 'fast'])
# OOP style object creation: make a figure object without axes
myfig = plt.figure()
# function subplots creates a figure object and an axes grid
# plot some data on the axes with axes method plot()
fig1, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
# equivalently: pyplot style function call 
# plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
# now make a figure with a 2x2 grid of axes
fig, axs = plt.subplots(2,2)
fig.set_size_inches(14., 10.)
# plot some data on each axes
x = np.linspace(0, 2, 100)
axs[0,0].plot(x, x, label='linear')
axs[0,0].set_title("axes[0,0]")
axs[0,0].set_xlabel('x')
axs[0,0].set_ylabel('y')
axs[0,0].legend()
axs[0,1].plot(x, x**2, label='quadratic')
axs[0,1].set_title("axes[0,1]")
axs[0,1].set_xlabel('x')
axs[0,1].set_ylabel('y')
axs[0,1].legend()
axs[1,0].plot(x, np.exp(x), label='exponential')
axs[1,0].set_title("axes[1,0]")
axs[1,0].set_xlabel('x')
axs[1,0].set_ylabel('y')
axs[1,0].legend()
axs[1,1].plot(x, np.log(x), label='logaritmic')
axs[1,1].set_title("axes[1,1]")
axs[1,1].set_xlabel('x')
axs[1,1].set_ylabel('y')
axs[1,1].legend()
# plotting data directly from a numpy recarray data object
data = {'a': np.arange(50),
        'c': np.random.randint(0, 50, 50),
        'd': np.random.randn(50)}
data['b'] = data['a'] + 10 * np.random.randn(50)
data['d'] = np.abs(data['d']) * 100
# make a new figure
fig2, ax2 = plt.subplots()
plt.scatter('a', 'b', c='c', s='d', data=data)
plt.xlabel('entry a')
plt.ylabel('entry b')
# save a specific figure to a file
print(fig.canvas.get_supported_filetypes())
fig.savefig('functions.png', transparent=False, dpi=80, bbox_inches="tight")
plt.show()
