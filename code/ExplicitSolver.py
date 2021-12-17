import numpy as np
import matplotlib.pyplot as plt


def exact(x, t):
    return np.sin(np.pi * x) * np.exp(-np.pi**2 * t)


def MSE(u, u_pred):
    return np.mean(0.5*(u - u_pred)**2)

def explicit_scheme(dx = 0.1, dt = 0.001, max_time = 1, tol = 1e-5):
    S = dt/dx**2
    if S > 0.5:
        print('Stability condition not fulfilled.')
        exit()

    L = 1
    Nx = int(L/dx) + 1                
    Nt = int(max_time/dt) + 1
 
    x = np.linspace(0, L, Nx )
    t = np.linspace(0, max_time, Nt )

    u_old = np.zeros(Nx)
    u_new = np.zeros(Nx)

    u_old[1:-1] = np.sin(np.pi*x[1:-1])
    u = np.zeros((Nt,u_old.shape[0]))
    
    for i in range(1, Nt):
        u_new[1:-1] = u_old[1:-1]*(1-2*S) + S*(u_old[2:] + u_old[:-2])

        
        u[i] = u_new

        u_old, u_new = u_new, u_old
        if u_new[1] < tol: # checks if the distribution is almost flat 
            return i, x, t, u

    return i, x, t, u

if __name__ == '__main__':
 
    dx = 0.01
    dt = 0.00001
    max_time = 1 # seconds
    tol = 1e-5

    idx, x, t, u_pred = explicit_scheme(dx, dt, max_time, tol)

    idxs = np.linspace(0,idx-1,5).astype('int')+1
    
    for i in idxs:
        u = exact(x,t[i-1])
        mse = MSE(u, u_pred[i])
        plt.plot(x,u, label=f'Analytic, t = {t[i-1]:.3f}s')
        plt.plot(x, u_pred[i], '.', label=f'Numerical, t = {t[i-1]:.3f}s, MSE = {mse:.3g}')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title(f'dx = {dx:g}, dt = {dt:g}, tol = {tol:g}')
    plt.legend()
    plt.show()
