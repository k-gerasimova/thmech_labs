import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp

def Rot2D(X,Y,Phi):
    RotX = X*np.cos(Phi) - Y*np.sin(Phi)
    RotY = X*np.sin(Phi) + Y*np.cos(Phi)
    return RotX, RotY


t = sp.Symbol('t')
r = 2 + sp.sin(8*t)
phi = t + 0.2 * sp.cos(6*t)
x = r*sp.cos(phi)
y = r*sp.sin(phi)
Vx = sp.diff(x, t)
Vy = sp.diff(y, t)
Wx = sp.diff(Vx, t)
Wy = sp.diff(Vy, t)
V = sp.sqrt(Vx * Vx + Vy * Vy)
tanx = Vx / V
tany = Vy / V
k = sp.diff(V, t)
W = sp.sqrt(Wx * Wx + Wy * Wy)
ksi = sp.acos((Vx * Wx + Vy * Wy) / (V * W))
Wnx = Wx - k * tanx
Wny = Wy - k * tany

F_x = sp.lambdify(t, x)
F_y = sp.lambdify(t, y)
F_Vx = sp.lambdify(t, Vx)
F_Vy = sp.lambdify(t, Vy)
F_Wx = sp.lambdify(t, Wx)
F_Wy = sp.lambdify(t, Wy)
F_Wnx = sp.lambdify(t, Wnx)
F_Wny = sp.lambdify(t, Wny)


T = np.linspace(0, 10, 1001)
X = np.zeros_like(T)
Y = np.zeros_like(T)
VX = np.zeros_like(T)
VY = np.zeros_like(T)
WX = np.zeros_like(T)
WY = np.zeros_like(T)
WNX = np.zeros_like(T)
WNY = np.zeros_like(T)


for i in np.arange(len(T)):
    X[i] = F_x(T[i])
    Y[i] = F_y(T[i])
    VX[i] = F_Vx(T[i])
    VY[i] = F_Vy(T[i])
    WX[i] = F_Wx(T[i])
    WY[i] = F_Wy(T[i])
    WNX[i] = F_Wnx(T[i])
    WNY[i] = F_Wny(T[i])


Phi = np.arctan2(VY, VX)
Psi = np.arctan2(WY, WX)
fig = plt.figure(figsize=[13,8]) #создаём фигуру
ax = fig.add_subplot(1,1,1) #создаём оси
ax.axis('equal')
ax.set(xlim=[-(max(X) - min(X))/2+min(X), (max(X) - min(X))/2+max(X)],
ylim=[-(max(Y) - min(Y))/2+min(Y), (max(Y) - min(Y))/2+max(Y)])

ax.plot(X, Y) #рисуем

P = ax.plot(X[0], Y[0], marker='o')[0]
Ro = ax.plot(X[0] + WNX[0], Y[0] + WNY[0], marker='p')[0]
V_Line = ax.plot([X[0], X[0]+VX[0]], [Y[0], Y[0]+VY[0]])[0]
W_Line = ax.plot([X[0], X[0]+WX[0]], [Y[0], Y[0]+WY[0]])[0]
Wnline = ax.plot([X[0], X[0] + WNX[0]], [Y[0], Y[0] + WNY[0]])[0]

XArrow = np.array([-0.15, 0, -0.15])
YArrow = np.array([0.1, 0, -0.1])

RArrowX, RarrowY = Rot2D(XArrow, YArrow, Phi[0])
WArrowX, WarrowY = Rot2D(XArrow, YArrow, Psi[0])
V_Arrow = ax.plot(X[0]+VX[0]+RArrowX, Y[0]+VY[0]+RarrowY)[0]
W_Arrow = ax.plot(X[0]+WX[0]+RArrowX, Y[0]+VY[0]+RarrowY)[0]
WNArrow = ax.plot(X[0] + WNX[0],Y[0] + WNY[0])[0]


def MagicOfTheMovement(i):
    P.set_data(X[i], Y[i])
    V_Line.set_data([X[i], X[i]+VX[i]], [Y[i], Y[i]+VY[i]])
    W_Line.set_data([X[i], X[i] + WX[i]], [Y[i], Y[i] + WY[i]])
    Wnline.set_data([X[i], X[i] + WNX[i]], [Y[i], Y[i] + WNY[i]])
    Ro.set_data(X[i] + WNX[i], Y[i] + WNY[i])
    RArrowX, RarrowY = Rot2D(XArrow, YArrow, Phi[i])
    WArrowX, WarrowY = Rot2D(XArrow, YArrow, Psi[i])
    V_Arrow.set_data(X[i]+VX[i]+RArrowX, Y[i]+VY[i]+RarrowY)
    W_Arrow.set_data(X[i] + WX[i] + WArrowX, Y[i] + WY[i] + WarrowY)
    return [P, V_Line, V_Arrow, W_Line, W_Arrow, Wnline, Ro]

nechto = FuncAnimation(fig, MagicOfTheMovement, interval=20, frames=len(T))
plt.show() #просим его это показать
