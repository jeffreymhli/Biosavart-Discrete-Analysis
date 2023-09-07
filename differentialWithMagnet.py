import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math

const1 = {
    "c": 46.263333333333335,
    "d": 0.0682,
    #"d": 0.06623693,
    #"d": 0.06476115,

    "m": 0.05545,
    "l": 0.0045,
    "r": 0.015,
    "u": 1.257*10**-6,
    "p": 0.017621365441726242,
    "lr": 0.501
}

const2 = {
    "c": 64.16,
    "d": 0.05,
    "b": 0.385782,

    "m": 0.0299,
    "l": 0.0045,
    "r": 0.015,
    "u": 1.257*10**-6,
    #"p": 0.008395362635583496
    "p": 0.039391269,
    #"p": 0.0337450623279917
    "lr": 0.1415
}

const3 = {
    "c": 64.16,
    #"d": 0.06974101,
    #"d": 0.09800044,

    "m": 0.0299,
    "l": 0.0045,
    "r": 0.015,
    "u": 1.257*10**-6,
    "p": 0.022373610272,
    "lr": 0.1415
}

numCoils = 10 #number of loops in model
numSum = 10 #number of steps in integral
r = 0.015 #radius of loop/magnet
u = 1.257*10**-6 #constant: permeability of free space
i = 620000 #current
#i = 1000000

l = 0.0045 #length of magnet
hInc = 0
if not numCoils == 1:
    hInc = l / (numCoils - 1) #distance between coils
dtheta = 2*np.pi / numSum #step size of integral


def integrali(theta, x, y, z, i):
    #integrate x component
    return (u*i / (4*np.pi))*(r*z*math.cos(theta)) / (x**2 + y**2 + z**2 + r**2 -2*r*(x*math.cos(theta) + y*math.sin(theta)))**(1.5)

def integralj(theta, x, y, z, i):
    #integrate y component
    return (u*i / (4*np.pi))*(r*z*math.sin(theta)) / (x**2 + y**2 + z**2 + r**2 -2*r*(x*math.cos(theta) + y*math.sin(theta)))**(1.5)

def integralk(theta, x, y, z, i):
    #integrate z component
    return -(u*i / (4*np.pi))*(y*math.sin(theta) + x*math.cos(theta) - r) * r / (x**2 + y**2 + z**2 + r**2 -2*r*(x*math.cos(theta) + y*math.sin(theta)))**(1.5)

def findField(coils, h, x, y, z, i):
    sumi = 0
    sumj = 0
    sumk = 0
    for n in range(coils):
        forcei = forcej = forcek = 0
        for q in range(numSum):
            angle = q*dtheta
            forcei += dtheta*integrali(angle, x, y, z - n*h, i)
            forcej += dtheta*integralj(angle, x, y, z - n*h, i)
            forcek += dtheta*integralk(angle, x, y, z - n*h, i)

        sumi += h*forcei
        sumj += h*forcej
        sumk += h*forcek
    
    return [sumi, sumj, sumk]

def crossProd(vec1, vec2):
    return [vec1[1]*vec2[2] - vec1[2]*vec2[1], vec1[2]*vec2[0] - vec1[0]*vec2[2], vec1[0]*vec2[1] - vec1[1]*vec2[0]]

def findForce(theta, alpha, r, x, y, z, hInc, layer, coils, i):
    increment = hInc*layer
    sa, ca, st, ct = math.sin(alpha), math.cos(alpha), math.sin(theta), math.cos(theta)
    fieldVec = findField(coils, hInc, x - r*st*ca + increment*sa, y + r*ct, z + r*st*sa + increment*ca, i)
    dl = [r*ct*ca, st*r, -ct*sa*r]
    return [q*i for q in crossProd(dl, fieldVec)]

def forceonCoil(coils, alpha, x, y, z, r):

    sumi = sumj = sumk = 0

    for n in range(coils):
        forcei = forcej = forcek = 0

        for q in range(numSum):
            angle = q*dtheta
            foundForce = findForce(angle, alpha, r, x, y, z, hInc, -n, coils, i)
            forcei += dtheta*foundForce[0]
            forcej += dtheta*foundForce[1]
            forcek += dtheta*foundForce[2]

        sumi += hInc*forcei
        sumj += hInc*forcej
        sumk += hInc*forcek


    return [sumi, sumj, sumk]

def xpos(l, theta):
    return l*(1-math.sin(theta)) / (np.pi/2 - theta)

def ypos(l, theta):
    return l*math.cos(theta) / (np.pi/2 - theta)

def interpolatePos(a1, a2, d):
    deltax = d + xpos(const1["lr"] - r, a1) + xpos(const1["lr"] - r, a2)
    deltay = abs(ypos(const1["lr"] - r, a1) - ypos(const1["lr"] - r, a2))
    #print(deltax, deltay)

    relx = deltax*math.cos(a1) - deltay*math.sin(a1)
    rely = deltax*math.sin(a1) + deltay*math.cos(a1)
    reltheta = np.pi - a1 - a2
    unaltered = forceonCoil(numCoils, reltheta, relx, 0, rely, r)
    altered = [unaltered[0]*math.cos(a1) + unaltered[2]*math.sin(a1), unaltered[1], unaltered[2]*math.cos(a1) - unaltered[0]*math.sin(a1)]
    return altered

def totalForce(a1, a2, d):
    indivForces = interpolatePos(a1, a2, d)
    return math.sqrt((indivForces[0])**2 + (indivForces[1])**2 + (indivForces[2])**2)

#print(interpolatePos(np.pi/2-0.0001, np.pi/2+0.0001, const1["d"]))
def newtonsMethod(depth, maxdepth, f, df, current):
    if depth == maxdepth:
        return current
    else:
        return newtonsMethod(depth+1, maxdepth, f, df, current - f(current) / df(current))

#print(newtonsMethod(0, 20, lambda x: x**2 - 4, lambda x: 2*x, np.pi/2-0.001))

def odes(x,t):
    #x1,x2,pos1,pos2 = x
    y1, y2, y3, y4 = x
    
    a1 = newtonsMethod(0,10,lambda x: (const1["lr"]-r)*(1 - math.sin(x)) / (np.pi/2 - x) + y3, lambda x: (const1["lr"]-r)*((-math.cos(x)*(np.pi/2 - x) + (1 - math.sin(x)))) / (np.pi/2 - x)**2,np.pi/2+0.001)
    a2 = newtonsMethod(0,10,lambda x: (const1["lr"]-r)*(1 - math.sin(x)) / (np.pi/2 - x) - y1, lambda x: (const1["lr"]-r)*((-math.cos(x)*(np.pi/2 - x) + (1 - math.sin(x)))) / (np.pi/2 - x)**2,np.pi/2+0.001)
    fm = interpolatePos(a1, a2, const1["d"])[0]
    #print(a1, a2, fm)
    dy1 = y2
    dy2 = - const1["p"] * y2 / const1["m"]  - const1["c"] * y1 / const1["m"] + fm / const1["m"]
    dy3 = y4
    dy4 = - const1["p"] * y4 / const1["m"]  - const1["c"] * y3 / const1["m"] - fm / const1["m"]
    
    #return [pos1,pos2,dx1dt2,dx2dt2]
    return [dy1,dy2,dy3,dy4]


# initial condition
#z0 = [0,0,-0.03806366,0]
z0 = [0.02801722,0,0,0]
#z0 = [0.05021156,0,0,0]
#z0 = [0.03566072,0,0,0]
#z0 = [0.02580301,0,0,0]
#z0 = [0.03285855,0,0,0]
#z0 = [0.03496397,0,0,0]
#z0 = [0.0277544,0,0,0]

#test defiend odes
#print(odes(z=x0,t=0))

#declare a time vector(time window)
t = np.linspace(0,15.4346875,3715)
#t = np.linspace(0,4.5007525,1071)
#t = np.linspace(0,13.177425,3172)
#t = np.linspace(0,13.6553125,3287)
#t = np.linspace(0,11.5530675,2781)
#t = np.linspace(0,7.99161,1924)
#t = np.linspace(0,15.62625,3761)
#t = np.linspace(0,11.53235,2776)
z = odeint(odes,z0,t)


x1 = z[:,0]
x2 = z[:,1]
x3 = z[:,2]
x4 = z[:,3]

for i in range(len(z[:,2])):
    print(str(z[:,2][i]))

h = x1+x3
plt.plot(t, x1, color='g', alpha=0.7, linewidth=0.6)
#plt.plot(t, x2, color='r', alpha = 0.7, linewidth = 0.6)
plt.plot(t, x3, color='r', alpha=0.7, linewidth=0.6)
#plt.plot(t, x4, color='y', alpha = 0.7, linewidth = 0.6)
plt.plot(t,h,color = 'b', alpha = 0.2,linewidth = 0.6 )
plt.show()
