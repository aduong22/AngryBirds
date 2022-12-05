from numpy import *
from matplotlib import pyplot
import matplotlib.animation as manimation
import os, sys

# HW5 Skeleton 

def RK4(f, y, t, dt, food_flag, alpha, gamma_1, gamma_2, kappa, rho, delta):
    '''
    Carry out a step of RK4 from time t using dt
    
    Input
    -----
    f:  Right hand side value function (this is the RHS function)
    y:  state vector
    t:  current time
    dt: time step size
    
    food_flag:  0 or 1, depending on the food location
    alpha:      Parameter from homework PDF
    gamma_1:    Parameter from homework PDF
    gamma_2:    Parameter from homework PDF
    kappa:      Parameter from homework PDF
    rho:        Parameter from homework PDF
    delta:      Parameter from homework PDF
    

    Output
    ------
    Return updated vector y, according to RK4 formula
    '''
    
    # Task: Fill in the RK4 formula
    k1 = f(y, t, food_flag, alpha, gamma_1, gamma_2, rho, delta)
    k2 = f(y+(dt*0.5)*k1, t+(dt*0.5), food_flag, alpha, gamma_1, gamma_2, kappa, rho, delta)
    k3 = f(y+(dt*0.5)*k2, t+(dt*0.5), food_flag, alpha, gamma_1, gamma_2, kappa, rho, delta)
    k4 = f(y+dt*k3, t+dt, food_flag, alpha, gamma_1, gamma_2, kappa, rho, delta)
    y = y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    return y


def RHS(y, t, food_flag, alpha, gamma_1, gamma_2, kappa, rho, delta):
    '''
    Define the right hand side of the ODE

    '''
    N = y.shape[0]
    f = zeros_like(y)
    
    # Task:  Fill this in by assigning values to f
    if (food_flag == 0):
        C = (0.0, 0.0)
    elif (food_flag == 1):
        C = (sin(alpha*t), cos(alpha*t))

    f_food = gamma_1*(C - y[0,:])
    f_follow = gamma_2*(y[0,:] - y)
    
    B_bar = sum(y, axis=0)/N
    f_fl = kappa*(B_bar - y)
    f_fl[0] = 0
    
    arr_dist = zeros((N,2))
    f_rep = zeros_like(y)
    for k in range(1,N):
        for i in range (0,N):
            arr_dist[i] = (i, sqrt((y[i,1] - y[k,1])**2 + (y[i,0] - y[k,0])**2))
        arr_d = argsort(arr_dist, axis = 0)
        
        if ((smelly_bird and k == 29) or (predator and k == 30)):
            continue
        for j in arr_d[1:6]:
            f_rep[k,:] += rho*((y[k,:] - y[j[1],:]) / ((y[k,:] - y[j[1],:])**2 + delta))
 
    f_sb = zeros_like(y)
    if smelly_bird:
        f_sb = rho*((y - y[29,:]) / ((y - y[29,:])**2 + delta))*3

    f_p = zeros_like(y)
    if predator:
        y[29,0] = -2
        y[29,1] = 2
        predator_loc = (y[29,0], y[29,1])
        f_p =  gamma_2*(y - predator_loc)/6
        f_p[30,:] = gamma_2*(y[0,:] - predator_loc)/5

    f = f_food + f_follow + f_fl + f_rep + f_sb + f_p
    
    return f


##
# Set up problem domain
t0 = 0.0        # start time
T = 10.0        # end time
nsteps = 75     # number of time steps

# Task:  Experiment with N, number of birds
N = 30

# Task:  Experiment with the problem parameters, and understand how each parameter affects the system
dt = (T - t0) / (nsteps-1.0)
gamma_1 = 2.0
gamma_2 = 8.0
alpha = 0.4
kappa = 4.0
rho = 2.0 
delta = 0.5
food_flag = 1   # food_flag == 0: C(x,y) = (0.0, 0.0)
                # food_flag == 1: C(x,y) = (sin(alpha*t), cos(alpha*t))

# Intialize problem
y = random.rand(N,2)  # This is the state vector of each Bird's position.  The k-th bird's position is (y[k,0], y[k,1])
flock_diam = zeros((nsteps,))

# smelly bird
smelly_bird = False
# predator
predator = False

# Write first movie frame
# - File name format: flock_frame_k.png  for k = 0, 1, ...
# - Note that the figure size is set so that there are an even number of pixels
#   in each dimension.  This can be important for converting to movie later 
FFMpegWriter = manimation.writers['ffmpeg']
writer = FFMpegWriter(fps=6)
fig = pyplot.figure(0)
pp, = pyplot.plot([],[], 'k+') 
rr, = pyplot.plot([],[], 'r+')
sb, = pyplot.plot([],[], 'g+')
pr, = pyplot.plot([],[], 'b+')
pyplot.xlabel(r'$X$', fontsize='large')
pyplot.ylabel(r'$Y$', fontsize='large')
pyplot.xlim(-5,5)       # you may need to adjust this, if your birds fly outside of this box!
pyplot.ylim(-5,5)       # you may need to adjust this, if your birds fly outside of this box!


# Begin writing movie frames
with writer.saving(fig, "flock_movie.mp4", dpi=1000):

    # First frame
    pp.set_data(y[1:,0], y[1:,1])
    rr.set_data(y[0,0], y[0,1])
    if smelly_bird:
        sb.set_data(y[29,0], y[29,1])
    if predator:
        pr.set_data(-2, 2)
    writer.grab_frame()

    t = t0
    for step in range(nsteps):
        
        # Task: Fill in the code for the next two lines 
        y = RK4(RHS, y, t, dt, food_flag, alpha, gamma_1, gamma_2, kappa, rho, delta)
        B_bar = sum(y, axis = 0)/N
        
        distance = 0
        if predator:
            for i in range(N-1):
                distance += sqrt((y[i,1] - B_bar[1])**2 + (y[i,0] - B_bar[0])**2)
            ave_dist = distance / (N-1)
        else:
            for i in range(N):
                distance += sqrt((y[i,1] - B_bar[1])**2 + (y[i,0] - B_bar[0])**2)
            ave_dist = distance / (N)
        flock_diam[step] = ave_dist*2
        t += dt
        
        # Movie frame
        pp.set_data(y[:,0], y[:,1]) 
        rr.set_data(y[0,0], y[0,1]) 
        if smelly_bird:
            sb.set_data(y[29,0], y[29,1])
        if predator:
            pr.set_data(y[30,0], y[30,1])
        writer.grab_frame()
        

# Task: Plot flock diameter
pyplot.cla()
pyplot.plot(arange(nsteps) + 1, flock_diam)
pyplot.show()

