"""elastic_rods.py -- Python translation of elasticRods.m (6.8410 HW1, Problem 3).

A closed loop of a naturally straight, isotropic Discrete Elastic Rod
(Bergou et al. 2008), integrated with symplectic Euler + fast projection.

Starter code: everything except the blocks marked  ### PROBLEM 3(x) ###  is
done for you.  Requires numpy, scipy and matplotlib.

Conventions (kept from the MATLAB code):
  * vertex arrays are (3, nSamples), one column per vertex;
  * edge i goes from vertex i to vertex i+1 (mod nSamples);  "R" = right
    (outgoing) edge of a vertex, "L" = left (incoming) edge;
  * bishopFrame is (3, 2, nSamples+1): columns are the frame vectors (u, v)
    on edge j; entry nSamples is the frame transported all the way around,
    so holonomy = frame[nSamples] vs frame[0].

Run:  python elastic_rods.py [bendModulus] [twistModulus] [totalTwist]
"""
import sys
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


def elastic_rods(bendModulus=1.0, twistModulus=1.0, totalTwist=np.pi,
                 nSteps=10000, plotEvery=40, show=True):
    nSamples = 100
    dt = 0.001

    def curveFunction(t):
        return np.vstack([np.cos(2 * np.pi * t),
                          np.sin(2 * np.pi * t),
                          0.3 * np.sin(4 * np.pi * t)])

    curveData = SimpleNamespace()
    curveData.verts = curveFunction(np.linspace(0, 1, nSamples + 1))[:, :nSamples]
    curveData.totalTwist = totalTwist
    curveData.velocities = np.zeros((3, nSamples))
    curveData = prepare(curveData)

    # Set up Bishop frame on the first edge
    u0 = np.cross(curveData.tangentsR[:, 0], [0.0, 0.0, 1.0])
    u0 = u0 / np.linalg.norm(u0)
    v0 = np.cross(curveData.tangentsR[:, 0], u0)
    v0 = v0 / np.linalg.norm(v0)
    curveData.bishopFrame = propagateBishopFrame(curveData, np.column_stack([u0, v0]), nSamples)
    curveData = updateMaterialFrame(curveData)

    # Sparsity pattern of the constraint gradient used by fastProjection:
    # row i (edge i) touches the 3 coordinates of vertex i and of vertex i+1.
    idx = np.arange(nSamples)
    DCii = np.repeat(idx, 6)
    DCjj = ((3 * idx)[:, None] + np.arange(6)[None, :]).ravel() % (3 * nSamples)

    if show:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        drawRod(ax, curveData)

    for i in range(nSteps):
        bendForce = computeBendForce(curveData)
        twistForce = computeTwistForce(curveData)
        totalForce = bendModulus * bendForce + twistModulus * twistForce
        totalAcceleration = totalForce / curveData.dualLengths

        verts, velocities = symplecticEuler(curveData.verts, curveData.velocities, totalAcceleration, dt)
        verts, velocities = fastProjection(curveData.verts, verts, curveData.dualLengths,
                                           curveData.edgeLengthsR, dt, nSamples, DCii, DCjj)
        newCurveData = SimpleNamespace(verts=verts, velocities=velocities)
        newCurveData = prepare(newCurveData)
        newCurveData = updateTwist(newCurveData, curveData, nSamples)
        newCurveData = updateMaterialFrame(newCurveData)
        curveData = newCurveData

        # Energies are available via computeEnergy(curveData, bendModulus, twistModulus)
        # -- record them here if you want to plot them over time (Problem 3(c)).

        if show and i % plotEvery == 0:
            ax.cla()
            drawRod(ax, curveData)
            plt.pause(0.001)

    if show:
        plt.ioff()
        plt.show()
    return curveData


def drawRod(ax, curveData):
    """patch(...) + quiver3(...) from the MATLAB code: closed polyline plus material vector u."""
    P = np.hstack([curveData.verts, curveData.verts[:, :1]])
    ax.plot(P[0], P[1], P[2], color="black", linewidth=3)
    m = curveData.midpointsR
    u = curveData.u[:, :-1]
    ax.quiver(m[0], m[1], m[2], u[0], u[1], u[2], color="red", linewidth=1, length=0.15, normalize=False)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_zlim(-1.2, 1.2)
    ax.set_axis_off()


def prepare(curveData):
    """Basic closed-curve quantities (edges, lengths, tangents, curvature binormals)."""
    v = curveData.verts
    curveData.edgesR = np.roll(v, -1, axis=1) - v
    curveData.edgesL = np.roll(curveData.edgesR, 1, axis=1)
    curveData.midpointsR = 0.5 * (np.roll(v, -1, axis=1) + v)
    curveData.edgeLengthsR = np.sqrt(np.sum(curveData.edgesR ** 2, axis=0))
    curveData.edgeLengthsL = np.roll(curveData.edgeLengthsR, 1)
    curveData.dualLengths = 0.5 * (curveData.edgeLengthsR + curveData.edgeLengthsL)
    curveData.totalLength = np.sum(curveData.dualLengths)

    curveData.tangentsR = curveData.edgesR / curveData.edgeLengthsR
    curveData.tangentsL = np.roll(curveData.tangentsR, 1, axis=1)
    curveData.curvatureBinormalsDenom = (curveData.edgeLengthsL * curveData.edgeLengthsR
                                         + np.sum(curveData.edgesL * curveData.edgesR, axis=0))
    curveData.curvatureBinormals = (2 * np.cross(curveData.edgesL, curveData.edgesR, axis=0)
                                    / curveData.curvatureBinormalsDenom)
    curveData.curvatureSquared = np.sum(curveData.curvatureBinormals ** 2, axis=0)
    return curveData


def computeEnergy(curveData, bendModulus, twistModulus):
    """Returns (totalEnergy, bendingEnergy, twistEnergy, kineticEnergy)."""
    bendingEnergy = bendModulus * np.sum(np.sum(curveData.curvatureBinormals ** 2, axis=0)
                                         / (2 * curveData.dualLengths))
    twistEnergy = twistModulus * curveData.totalTwist ** 2 / (2 * curveData.totalLength)
    kineticEnergy = 0.5 * np.sum(curveData.dualLengths * np.sum(curveData.velocities ** 2, axis=0))
    totalEnergy = bendingEnergy + twistEnergy + kineticEnergy
    return totalEnergy, bendingEnergy, twistEnergy, kineticEnergy


def propagateBishopFrame(curveData, startFrame, nSamples):
    ### PROBLEM 3(a) - YOUR CODE HERE TO PARALLEL TRANSPORT AROUND THE CURVE ###
    # parallelTransport[:, :, i] must be the rotation P_i taking the Bishop frame
    # of edge i-1 onto edge i (index 0 aligns the last edge with edge 0).
    parallelTransport = np.zeros((3, 3, nSamples))
    ### END HOMEWORK PROBLEM ###

    bishopFrame = np.zeros((3, 2, nSamples + 1))
    bishopFrame[:, :, 0] = startFrame
    for j in range(1, nSamples + 1):
        ptIdx = j % nSamples
        bishopFrame[:, :, j] = parallelTransport[:, :, ptIdx] @ bishopFrame[:, :, j - 1]
    return bishopFrame


def axang2rotm(axis, angle):
    """Rodrigues' formula (replacement for MATLAB's axang2rotm). Identity for a zero axis."""
    n = np.linalg.norm(axis)
    if n < 1e-14 or abs(angle) < 1e-14:
        return np.eye(3)
    k = axis / n
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def updateTwist(newCurveData, oldCurveData, nSamples):
    """Transport the reference frame in time and update the total twist by the holonomy change."""
    t_old = oldCurveData.tangentsR[:, 0]
    t_new = newCurveData.tangentsR[:, 0]
    timePtAxis = np.cross(t_old, t_new)
    timePtAngle = np.arctan2(np.linalg.norm(timePtAxis), np.dot(t_old, t_new))
    timePt = axang2rotm(timePtAxis, timePtAngle)

    newCurveData.bishopFrame = propagateBishopFrame(newCurveData, timePt @ oldCurveData.bishopFrame[:, :, 0], nSamples)

    holonomy = (timePt @ oldCurveData.bishopFrame[:, :, nSamples]).T @ newCurveData.bishopFrame[:, :, nSamples]
    holonomyAngle = np.arctan2(holonomy[1, 0], holonomy[0, 0])
    newCurveData.totalTwist = oldCurveData.totalTwist - holonomyAngle
    return newCurveData


def updateMaterialFrame(curveData):
    """Material frame (u, v) = Bishop frame rotated by the (uniformly distributed) twist."""
    cumTwist = (curveData.totalTwist
                * np.cumsum(np.concatenate([[0.0], np.roll(curveData.dualLengths, -1)]))
                / curveData.totalLength)
    b1 = curveData.bishopFrame[:, 0, :]
    b2 = curveData.bishopFrame[:, 1, :]
    curveData.u = np.cos(cumTwist) * b1 + np.sin(cumTwist) * b2
    curveData.v = np.cos(cumTwist) * b2 - np.sin(cumTwist) * b1
    return curveData


### PROBLEM 3(c) Part I - YOUR CODE HERE ###
def computeBendForce(curveData):
    bendForce = np.zeros(curveData.verts.shape)
    return bendForce
### END HOMEWORK PROBLEM ###


### PROBLEM 3(c) Part II - YOUR CODE HERE ###
def computeTwistForce(curveData):
    twistForce = np.zeros(curveData.verts.shape)
    return twistForce
### END HOMEWORK PROBLEM ###


def symplecticEuler(verts0, velocities0, totalAcceleration, dt):
    """Update velocities and positions via symplectic Euler integration."""
    velocities = velocities0 + dt * totalAcceleration
    verts = verts0 + dt * velocities
    return verts, velocities


def fastProjection(verts0, verts, mass, lengthsR, dt, nSamples, DCii, DCjj, maxIter=1000):
    """Project positions onto the inextensibility constraints |e_i|^2 = lengthsR_i^2."""
    massRep = np.repeat(mass, 3)                       # M = diag(massRep), flat index 3*i + k
    edgesR = np.roll(verts, -1, axis=1) - verts
    constraint = np.sum(edgesR ** 2, axis=0) - lengthsR ** 2
    it = 0
    while np.max(np.abs(constraint)) > 1e-10 and it < maxIter:
        vals = np.hstack([-edgesR.T, edgesR.T]).ravel()  # row i: [-e_i ; +e_i]
        constraintGrad = 2 * sp.csr_matrix((vals, (DCii, DCjj)), shape=(nSamples, 3 * nSamples))
        MinvDC = constraintGrad.T.multiply(1.0 / massRep[:, None]).tocsc()   # M^{-1} DC^T
        DCMinvDC = (constraintGrad @ MinvDC).tocsc()
        dLambda = spla.spsolve(DCMinvDC, constraint)
        dx = -np.asarray(MinvDC @ dLambda).reshape(nSamples, 3).T
        verts = verts + dx
        edgesR = np.roll(verts, -1, axis=1) - verts
        constraint = np.sum(edgesR ** 2, axis=0) - lengthsR ** 2
        it += 1
    velocities = (verts - verts0) / dt
    return verts, velocities


if __name__ == "__main__":
    args = [float(a) for a in sys.argv[1:4]]
    elastic_rods(*args)
