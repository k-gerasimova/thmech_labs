import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

def odesys(y, t, M, a, k, b):
    #y[0] = tetta, y[1] = phi, y[2] = tetta', y[3] = phi'
    # yt[0] = tetta', yt[1] = phi', yt[2] = tetta'', yt[3] = phi''

    yt = np.zeros_like(y)
    yt[0] = y[2]
    yt[1] = y[3]

    a11 = M
    a12 = 0
    a21 = 0
    a22 = M*(a**2 + (b**2*(np.sin(y[0]))**2))

    b1 = -k * y[2] +M*y[3]**2*np.sin(y[0])*np.cos(y[0])
    b2 = -k*y[3]*(a**2+b**2*(np.sin(y[0]))**2)- y[3]*M*y[2]*b**2*np.sin(2*y[0])

    yt[2] = (b1*a22 - a12*b2)/(a11*a22 - a12*a21)
    yt[3] = (b2 * a11 - a21 * b1) / (a11 * a22 - a12 * a21)
    return yt



M = 5
a = 1
b = 0.25
k = 9
t0 = 0
phi0 = 0
tetta0 = 0
dphi0 = 5
dtetta0 =0

y0 = [tetta0, phi0, dtetta0, dphi0]

Tfin = 10
Tcol = 1001

t = np.linspace(t0, Tfin, Tcol)

Y = odeint(odesys, y0, t, (M, a, k, b))



fig=plt.figure(figsize=[13,9])
ax = fig.add_subplot(projection='3d')   # добавился аргумент

tetta = Y[:,0]
phi = Y[:,1]
dtetta = Y[:,2]
dphi = Y[:,3]
ddtetta = [odesys(y0, t, M, a, k, b)[2] for y0, t in zip(Y, t)]
ddphi = [odesys(y0, t, M, a, k, b)[3] for y0, t in zip(Y, t)]

result = [num * M for num in ddphi]
N = 2*a*(result+k*dphi)
fig_for_graphs = plt.figure(figsize=[13,13])
ax_for_graphs = fig_for_graphs.add_subplot(3,1,1)
ax_for_graphs.plot(t,tetta,color='yellow')
ax_for_graphs.set_title("tetta(t)")
ax_for_graphs.set(xlim=[0,Tfin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(3,1,2)
ax_for_graphs.plot(t,phi,color='orange')
ax_for_graphs.set_title('phi(t)')
ax_for_graphs.set(xlim=[0,Tfin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(3,1,3)
ax_for_graphs.plot(t,N,color='green')
ax_for_graphs.set_title('N(t)')
ax_for_graphs.set(xlim=[0,Tfin])
ax_for_graphs.grid(True)

#tetta = np.arange(0, 100 * np.pi, 0.1)
#phi = np.zeros_like(tetta)
#phi = np.linspace(0, 10 * np.pi, len(tetta))
x, y, z = a * np.cos(phi), a * np.sin(phi), b*np.cos(tetta)
X_A = (a - b * np.sin(tetta)) * np.cos(phi)
Z_A = b*np.cos(tetta)
Y_A = (a - b * np.sin(tetta)) * np.sin(phi)
X_B = (a + b * np.sin(tetta)) * np.cos(phi)
Z_B = -Z_A
Y_B = (a + b * np.sin(tetta)) * np.sin(phi)


#Spiral=ax.plot(x, y, max(z)/2)[0]   # просто добавляем 3 координату
#SSpiral =ax.plot(X_A,  Y_A , max(z)/2+Z_A)[0]
Vertical=ax.plot([0, 0], [0, 0], [min(z)-0.5, max(z)+0.5],color=[0, 0, 0],linestyle='dashed')[0]
YHorizontal = ax.plot([0, 0], [min(y)-1, max(y)+1], [max(z)/2 , max(z)/2],color=[0, 0, 0],linestyle='dashed')[0]
HXHorizontal = ax.plot([min(x)-1, max(x)+1], [0,0], [max(z)/2 , max(z)/2],color=[0, 0, 0],linestyle='dashed')[0]

C = ax.plot(x[0],y[0],max(z)/2,color=[1, 0, 0],marker='o',markersize=10)[0]
D = ax.plot(0,0,max(z)/2,color=[0, 1, 0],marker='o')[0]
A = ax.plot(X_A[0], Y_A[0], max(z)/2+Z_A[0], color = [0,1,0], marker='o')[0]
B = ax.plot(X_B[0], Y_B[0], max(z)/2+Z_B[0], color = [0,1,0], marker='o')[0]
CD = ax.plot([0, x[0]],[0, y[0]],[max(z)/2, max(z)/2],color=[0, 0, 0])[0]
AC = ax.plot([x[0], X_A[0]], [y[0], Y_A[0]], [max(z)/2, max(z)/2+Z_A[0]], color =[0,0,0])[0]
AB = ax.plot([X_B[0], X_A[0]], [Y_B[0], Y_A[0]], [max(z)/2+Z_B[0], max(z)/2+Z_A[0]], color =[0,0,0])[0]

dt=0.01
def NewPoints(i):
    C.set_data_3d(x[i], y[i], max(z)/2)   # set_data_3d. добавили 3д.
    D.set_data_3d(0, 0, max(z)/2)
    A.set_data_3d(X_A[i], Y_A[i], max(z)/2+Z_A[i])
    B.set_data_3d(X_B[i], Y_B[i], max(z) / 2 + Z_B[i])
    CD.set_data_3d([0, x[i]], [0, y[i]], [max(z)/2, max(z)/2])
    AC.set_data_3d([x[i], X_A[i]], [y[i], Y_A[i]], [max(z) / 2, max(z) / 2 + Z_A[i]])
    AB.set_data_3d([X_B[i], X_A[i]], [Y_B[i], Y_A[i]], [max(z) / 2 + Z_B[i], max(z) / 2 + Z_A[i]])
    return [D,C,CD, A, AC, AB, B]
   # return [AC]

a = FuncAnimation(fig, NewPoints, interval=dt*1000, blit=True, frames=len(x))




plt.show()
