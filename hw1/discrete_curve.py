"""discrete_curve.py -- Python translation of discreteCurve.m (6.8410 HW1, Problem 2).

Starter code: visualization and problem setup are done for you; fill in the
blocks marked  ### YOUR CODE ... ###.  Requires numpy and matplotlib.

Run:  python discrete_curve.py
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- Problem 2(c)
a, b = 4, 2
delta = np.pi / 3
n = 100

t = np.linspace(0, 2 * np.pi, n)
x = np.sin(a * t * delta)
y = np.sin(b * t)

u = np.zeros(n - 2)          # x-component of the gradient at interior vertices
v = np.zeros(n - 2)          # y-component of the gradient at interior vertices
xy = np.vstack([x, y])       # shape (2, n), one column per vertex
diff = xy[:, 1:] - xy[:, :-1]  # shape (2, n-1), edge vectors x_{i+1} - x_i

### YOUR CODE TO COMPUTE GRADIENT HERE ###
diff_normalized = diff / np.linalg.norm(diff, axis=0)
tan_xy = diff_normalized [:, :-1] + diff_normalized[:, 1:]
u, v = tan_xy [0, :], tan_xy [1, :]
### END HOMEWORK PROBLEM ###

plt.figure()
plt.plot(x, y, linewidth=2, color="black")
plt.quiver(x[1:-1], y[1:-1], u, v, linewidth=1, color="red")  # autoscaled like MATLAB's quiver
plt.axis("equal")
plt.title("Problem 2(c): gradient of arc length at each vertex")

# ---------------------------------------------------------------- Problem 2(d)
kappa = np.zeros(n - 2)      # per-vertex (unsigned) discrete curvature

### YOUR CODE TO COMPUTE KAPPA HERE ###
u = diff_normalized[:, :-1] # following the convention in the notes
v = -diff_normalized[:, 1:] # following the contention in the notes
variation = u + v

kappa_denom = np.linalg.norm(variation , axis=0)     # per-vertex (unsigned) discrete curvature
L = np.linalg.norm(diff, axis=0)
kappa = 2 * kappa_denom / (L[:-1] + L[:1])
### END HOMEWORK PROBLEM ###

# Curve colored by kappa (MATLAB's surface/'edgecolor','interp' trick -> LineCollection)
from matplotlib.collections import LineCollection

X = x[1:-1]
Y = y[1:-1]
points = np.column_stack([X, Y]).reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
seg_vals = 0.5 * (kappa[:-1] + kappa[1:])
fig, ax = plt.subplots()
lc = LineCollection(segments, cmap="viridis", linewidth=2)
lc.set_array(seg_vals)
ax.add_collection(lc)
ax.autoscale()
ax.set_aspect("equal")
fig.colorbar(lc, ax=ax)
ax.set_title("Problem 2(d): curve colored by discrete curvature")

# ---------------------------------------------------------------- Problem 2(e)
t0 = 0.0
t1 = np.pi * 1.25
nSamples = 100
nSteps = 2000
h = 0.005                     # step size: the largest that keeps the length decreasing for all 2000 steps (0.01 wobbles after step 820)
drawEvery = 20                # redraw (and keep a GIF frame) every drawEvery steps

# We provide a few examples of curves to try (each returns an (nSamples, 2) array)
# curveFunction = lambda t: np.column_stack([np.cos(t) - np.cos(3 * t) ** 3, np.sin(t) - np.sin(3 * t) ** 3])
# curveFunction = lambda t: np.column_stack([np.cos(t), np.sin(t)])
curveFunction = lambda t: np.column_stack([t, (t - t0) * (t1 - t)])
curve = curveFunction(np.linspace(t0, t1, nSamples))

# Time step
plt.ion()
fig = plt.figure()
(plt_line,) = plt.plot(curve[:, 0], curve[:, 1], "k", linewidth=2)
plt.axis("equal")
plt.title("Problem 2(e): curve-shortening by gradient descent")
frames = [curve.copy()]      # one copy per drawn step so the animation can be saved after the loop
for i in range(nSteps):
    ### YOUR CODE HERE TO PERFORM GRADIENT DESCENT ###
    diff = curve[1:] - curve[:-1]                                 # (n-1, 2) edge vectors x_{i+1} - x_i
    diff_normalized = diff / np.linalg.norm(diff, axis=1, keepdims=True)
    u = diff_normalized[:-1]     # following the convention in the notes
    v = -diff_normalized[1:]     # following the convention in the notes
    variation = u + v            # (n-2, 2) gradient of s at the interior vertices
    curve[1:-1] -= h * variation
    ### END HOMEWORK PROBLEM ###
    if (i + 1) % drawEvery == 0:
        plt_line.set_xdata(curve[:, 0])
        plt_line.set_ydata(curve[:, 1])
        fig.canvas.draw()
        plt.pause(0.05)
        frames.append(curve.copy())

# discreteCurve.m only animates the figure window with drawnow. When this script runs
# without a display (MPLBACKEND=Agg) nothing is shown, so the same frames are also saved as a GIF.
import os
from matplotlib.animation import FuncAnimation, PillowWriter

os.makedirs("outputs", exist_ok=True)
def _draw(k):
    plt_line.set_data(frames[k][:, 0], frames[k][:, 1])
    return (plt_line,)
anim = FuncAnimation(fig, _draw, frames=len(frames), blit=True)
anim.save("outputs/discrete_curve-p2e-shortening.gif", writer=PillowWriter(fps=12))   # 101 frames, about 8 s

plt.ioff()
plt.show()
