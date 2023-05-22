# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 11:08:08 2023

@author: jonas
"""

# %% imports
import datetime
from random import uniform as rand
import sys
import os
#from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
import time
from functools import cache, lru_cache
from scipy import optimize as opt,linalg
import matplotlib.pyplot as plt
from scipy.integrate import quad, _quadpack, dblquad, tplquad, trapezoid as td
#from numpy import sqrt, sin, cos, pi, exp, inf,sinh,cosh, log, heaviside, tanh
from mpmath import sqrt, sin, cos, pi, exp, inf,sinh,cosh, log, tanh
import numpy as np
from mpmath import mp
mp.dps = 15
mp.pretty = True
plt.close("all")

# %% Parameter

def prec(x):
    return mp.mpf(str(x))

# Parameters '1' is mechanics, '2' is spin
T = prec(27*10**(-6)) #[s]
zeta1 = prec(-0.3)
zeta2 = prec(0.03)
r1 = prec(-0.7*2*pi*20*10**3) #[Hz]
r2 = prec(2*pi*20*10**3) #[Hz]
Gamma10 = prec(2*2*pi*20*10**3) #[Hz]
G20Max = prec(3*2*pi*20*10**3) #[Hz]
Gamma20 = prec(exp(-r2*T)*G20Max) #[Hz] 
gamma10 = prec(2*pi*2.1*10**(-3)) #[Hz]
gamma20 = prec(0.85)/prec(20)*Gamma20 #[Hz]
eta = prec(0.95)
nu = prec(0.75)
n1 = prec(173*10**3)
n2 = 0
nbar = 4
nbar1 = 0
M1 = None#1
M2 = None#2*nbar/(2*nbar+1)
M = 1
# gammad1 = gamma1*(n1+1/2) #[Hz]
# gammad2 = gamma2*(n2+1/2) #[HZ]

def dimesionlessParameters(T):
    p = {
          "T":T, 
          "zeta1":zeta1,"zeta2":zeta2,"Gamma10":Gamma10*T,"Gamma20":Gamma20*T,
          "gamma10":gamma10*T,"gamma20":gamma20*T,"eta":eta,"nu":nu,"n1":n1,"n2":n2,
          "M1":M1,"M2":M2,"M":M, "nbar":nbar, "nbar1":nbar1,"r1":r1*T,"r2":r2*T,
          "G20Max":G20Max*T
        }
    return p

p = dimesionlessParameters(T)


# All time arguments are scaled by 1/T and other parameters are scaled
# appropriately so that all functions are dimensionless
# %% Functions

##   Fitler in the ideal case:
def fv(t):
    return p["zeta1"]*p["M"]*sqrt(G.G2(t))*exp(-p["zeta1"]*G.G2(t)*(1-t))\
            /sinh(p["zeta1"]*G.G2(t)*1)
        
class Gamma_pars:
    
    def __init__(self,A=0.5*p["Gamma10"],a=p["r1"],
                 B=0.5*p["Gamma10"],b=-p["r1"]):

        self.A,self.a,self.B,self.b = prec(A),prec(a),prec(B),prec(b)
        
    def __str__(self):
        return "A*exp(a*t)+B*exp(b*t)"
    
    @lru_cache(maxsize=int(1e5))
    def G1(self,t):
        t = prec(t)
        return self.A*exp(self.a*t)+self.B*exp(self.b*t)
    
    @lru_cache(maxsize=int(1))
    def G2(self,t):
        return p["G20Max"] 


class Gamma_scalar:
    
    def __init__(self,r1=p["r1"]):

        self.r1 = prec(r1) 
    
    def __str__(self):
        return "Gamma10*exp(r1*t)"
    
    @lru_cache(maxsize=int(1e5))
    def G1(self,t):
        t = prec(t)
        return p["Gamma10"]*exp(self.r1*t)
    
    @lru_cache(maxsize=int(1))
    def G2(self,t):
        return p["G20Max"] 
    
class Gamma_exp2:
    
    def __init__(self,r1=p["r1"],r2=p["r2"]):
        
        self.r1 = prec(r1)
        self.r2 = prec(r2) 
    
    def __str__(self):
        return "G1=Gamma10*exp(r1*t),G2=G20Max*exp(r2*(t-1))"
    
    @lru_cache(maxsize=int(1e5))
    def G1(self,t):
        t = prec(t)
        return p["Gamma10"]*exp(self.r1*t)
    
    @lru_cache(maxsize=int(1e5))
    def G2(self,t):
        t = prec(t)
        return p["G20Max"]*exp(self.r2*(t-1))
        

class Gamma_sinh:
    
    def __init__(self,a=p["r1"]):

        self.a = prec(a) 
    
    def __str__(self):
        return "sinh(a*(1-x))^2/(sinh(a)^2*(1+1/Gamma10)-sinh(a*(1-x)^2)"
    
    @lru_cache(maxsize=int(1e5))
    def G1(self,t):
        t = prec(t)
        v1 = sinh(self.a*(1-t))
        v11 = v1*v1
        v2 = sinh(self.a)
        v22 = v2*v2*(1+1/p["Gamma10"])
        return v11/(v22-v11)
    
    @lru_cache(maxsize=int(1))
    def G2(self,t):
        return p["G20Max"] 


G = None#Gamma_scalar()

@lru_cache(maxsize=int(1e5))
def gamma1(t):
    t = prec(t)
    #return p["gamma10"]*exp(p["r1"]*t)
    return p["gamma10"]


@lru_cache(maxsize=int(1e5))
def gamma2(t):
    t = prec(t)
    #return p["gamma20"]*exp(p["r2"]*t)
    return p["gamma20"]



@lru_cache(maxsize=int(1e5))
def gammad1(t):
    t = prec(t)
    return gamma1(t)*(n1+1/2)


@lru_cache(maxsize=int(1e5))
def gammad2(t):
    t = prec(t)
    return gamma2(t)*(n2+1/2)


def gd1(t):
    t = prec(t)
    return gammad1(t)


def gd2(t):
    t = prec(t)
    return gammad2(t)


@lru_cache(maxsize=int(1e5))
def intgc1(t1,t2):
    t1 = prec(t1)
    t2 = prec(t2)
    return inte.integrate(lambda t: gamma1(t)+p["zeta1"]*G.G1(t),t1,t2)
    #return gamma1(0)*(t2-t1)+p["zeta1"]*inte.integrate(G.G1,t1,t2)


#Defing \int_{t_1}^{t_2} dx\check{\gamma}_{2v}(x):
@lru_cache(maxsize=int(1e5))
def intgc2(t1,t2):
    t1 = prec(t1)
    t2 = prec(t2)
    return inte.integrate(lambda t: gamma2(t)-p["zeta1"]*G.G2(t),t1,t2)
    #return  gamma2(0)*(t2-t1)-p["zeta1"]*inte.integrate(Gamma2,t1,t2)


def heavi(x):
    #return heaviside(x,1/2)
    eps = prec(0.0009)
    x = prec(x)
    return prec((tanh(x/eps)+1)/2)

  
 #----------------------------------------------------------------------------   
 
    
class integration:
    
    def __init__(self,method=None):
        self.method = method

    def integrate(self,func,a,b):        
        if self.method == "mp-quad":
            return mp.quad(func,[a,b])    

inte = integration(method="mp-quad")

def RefArray(fv,x):
    
    thr0 = 30/100
    thr1 = 10/100
    thr2 = 100#1/100
    thr3 = 100#0.1/100
    xp = np.array([])
    
    def insert(x,i,n):
        
        return np.linspace(float(x[i-1]), float(x[i+1]),n+2)
    
    for i in range(1,len(x)-2):
        
        if abs((fv[i]-fv[i+1])/fv[i])>=thr0:
            xp=np.append(xp,insert(x,i,3))
        
        elif abs((fv[i]-fv[i+1])/fv[i])>=thr1:
            xp=np.append(xp,insert(x,i,2))
            
        elif abs((fv[i]-fv[i+1])/fv[i])>=thr2:
            xp=np.append(xp,insert(x,i,2))
        
        elif abs((fv[i]-fv[i+1])/fv[i])>=thr3:
            xp=np.append(xp,insert(x,i,1))
        
        else: pass

    return np.sort(np.unique(np.append(xp,x)))

##########################################################################
  
@lru_cache(maxsize=int(1e5))
def A1(t):
    t = prec(t)
    res = inte.integrate(lambda tp: fv(tp)*sqrt(G.G1(tp))*exp(intgc1(tp,t)),0,t)
    return res

@lru_cache(maxsize=int(1e5))
def A2(t):
    t = prec(t)
    res = inte.integrate(lambda tp: fv(tp)*sqrt(G.G2(tp))*exp(intgc2(tp,t)),t,1)
    return res
 
@lru_cache(maxsize=int(1e5))
def A1p(t,s):
    t = prec(t)
    s = prec(s)
    res = heavi(t-s)*sqrt(G.G1(s))*exp(intgc1(s,t))
    return res

@lru_cache(maxsize=int(1e5))
def A2p(t,s):
    t = prec(t)
    s = prec(s)
    res = heavi(s-t)*sqrt(G.G2(s))*exp(-intgc2(t,s))
    return res

@lru_cache(maxsize=int(1e5))   
def A10p(s):
    s = prec(s)
    res = sqrt(G.G1(s))*exp(intgc1(s,1))
    return res

@lru_cache(maxsize=int(1e5))
def A20p(s):
    s = prec(s)
    res = sqrt(G.G2(s))*exp(-intgc2(0,s))
    return res

@lru_cache(maxsize=int(1e5))
def A1b(t,s):
    t = prec(t)
    s = prec(s)
    return sqrt(G.G1(s))*exp(intgc1(s,t))

@lru_cache(maxsize=int(1e5))
def A2b(t,s):
    t = prec(t)
    s = prec(s)
    return sqrt(G.G2(s))*exp(-intgc2(t,s))

@lru_cache(maxsize=int(1e5))
def A12(t,s):
    t = prec(t)
    s = prec(s)
    res = p["zeta1"]*(sqrt(G.G1(t))*(A1p(t,s)-A1p(1,s)*exp(-intgc1(t,1)))+sqrt(G.G2(t))*A2p(t,s))
    return res

@lru_cache(maxsize=int(1e5))
def BXp(t,s):
    t = prec(t)
    s = prec(s)
    res = sqrt(G.G2(t))*A2p(t,s)-sqrt(G.G1(t))*A1p(t,s)
    return res

@lru_cache(maxsize=int(1e5))
def C1p(t,s):
    t = prec(t)
    s = prec(s)
    res = sqrt(nu)*(BXp(t,s)-A10p(s)*sqrt(G.G1(t))*exp(-intgc1(t,1)))
    return res

@lru_cache(maxsize=int(1e5))
def D1p(t,s):
    t = prec(t)
    s = prec(s)
    res = sqrt(nu)*(A1p(t,s)-A10p(s)*exp(-intgc1(t,1)))
    return res
    
    
@lru_cache(maxsize=1) 
def Nt():
    v1 = 1+p["zeta1"]*p["zeta1"]
    return p["nbar"]-1/2+(p["nbar1"]+1/2)*exp(-2*intgc1(0,1))\
           +inte.integrate(lambda t:(1/2*v1*G.G1(t)+2*gd1(t))*exp(-2*intgc1(t,1)),0,1)

@lru_cache(maxsize=500) 
def C(s):
    s = prec(s)
    return (-2)*p["nbar"]*A20p(s)-2*sqrt(p["nu"])*(p["nbar1"]+1/2)*A10p(s)*exp(-2*intgc1(0,1))\
            +inte.integrate(lambda t:(sqrt(G.G1(t))*(C1p(t,s)+sqrt(p["nu"])*p["zeta1"]*A12(t,s))\
            +4*gd1(t)*D1p(t,s))*exp(-intgc1(t,1)),0,1)+p["zeta1"]*sqrt(p["nu"])*sqrt(G.G1(s))*exp(-intgc1(s,1))   
                
            

@lru_cache(maxsize=50000)        
def Ab(s,sp):
    sp = prec(sp)
    s = prec(s)
    v1 = 1+p["zeta1"]*p["zeta1"]
    v2 = (p["zeta1"]+p["zeta2"])*(p["zeta1"]+p["zeta2"])
    
    return p["nu"]*(p["nbar1"]+1/2)*exp(-2*intgc1(0,1))*A10p(s)*A10p(sp)+1/2*(2*p["nbar"]+1)*A20p(s)*A20p(sp)\
            +1/2*inte.integrate(lambda t:((1/p["eta"]-1)*v2*G.G2(t)+v1*(1-p["nu"])*G.G2(t)\
                         +4*gd2(t))*A2p(t,s)*A2p(t,sp)+C1p(t,s)*C1p(t,sp)+p["nu"]*A12(t,s)*A12(t,sp)\
                             +4*gd1(t)*D1p(t,s)*D1p(t,sp),0,1)\
        +1/2*(((1-p["nu"])*p["zeta1"]-(1/p["eta"]-1)*(p["zeta1"]+p["zeta2"]))*(sqrt(G.G2(s))*A2p(s,sp)+sqrt(G.G2(sp))*A2p(sp,s)))\
        +1/2*p["nu"]*(A12(s,sp)+A12(sp,s))
  
def clear_cache():
    for func in dir_cache:
        try:
            exec(f"{func}.cache_clear()")
        except:
            print(f'{func} not cleared')
            pass        

dir_main = dir()
dir_cache=[f for f in dir() if hasattr(eval(f), "cache_clear")]
dir_cache.append('G.G1');dir_cache.append('G.G2')

#--------------------------------------------------------------------------------
  
def find_fopt_sub(u,file=None,uarr=None):
    
    N = len(u)-1
    
    du =[(u[1]-u[0])/2]
    [du.append((u[i+1]-u[i-1])/2) for i in range(1,N)]
    du.append((u[N]-u[N-1])/2)

    Cvec = mp.matrix([prec(C(u[i])) for i in (range(0,N+1))])
 
    dumat = mp.zeros(N+1,N+1)
    for i in (range(0,N+1)):
        dumat[i,i] = prec(du[i])
        
    Abmat = mp.zeros(N+1)
    
    for i in (range(0,N+1)):
        Abmat[i,i] = prec(Ab(u[i],u[i]))
        for j in range(i+1,N+1):
            Abmat[i,j] = Abmat[j,i] = prec(Ab(u[i],u[j]))
            
    Amat = Abmat+1/(2*p["eta"])*dumat**-1
    
    
    if file != None:
        Amatnp = np.zeros((N+1,N+1))
        
        for i in (range(0,N+1)):
            Amatnp[i,i] = Amat[i,i]
            for j in range(i+1,N+1):
                Amatnp[i,j] = Amatnp[j,i] = Amat[i,j]
              
    b = -1/2*Cvec
    fvvec = mp.lu_solve(Amat, b)
    
    Cvecfvvec = Cvec.T*fvvec
    
    return 1/2*Cvecfvvec[0]+Nt(),Cvecfvvec,fvvec

def find_fopt(u,file,uarr=None):
    
    Result,_,fvvec= find_fopt_sub(u,file,uarr)   
    F = 1/(1+Result)
    
    return fvvec,F
        
def plot_fv(fv,u,title,i=None,file=None,show=False):
    plt.figure()
    plt.title(title)
    plt.plot(u,fv,"b.",label = r"$f_{v}^{\ opt}$")
    plt.legend()
    plt.grid()
    if show:
        plt.show()
    else:
        plt.savefig(f"Figs/{file}/fv{i}")
        plt.close()          
    
    
def plot_F(F,n,title,file=None,show=False):
    plt.figure()
    plt.title(title)
    plt.plot(n,F,"bo",label = r"$Fidelity$")
    plt.legend()
    plt.grid()
    if show: plt.show()
    else:
        plt.savefig(f"Figs/{file}/Fidelity")
        plt.close()
        
def plot(File_name,iteration=0,plot_fv = True, plot_F = False):
    file,n = File_name, iteration
    
    if plot_F:
        file_F = f"Figs/{file}/Fidelity.csv"
        File_F = np.genfromtxt(file_F,delimiter = ",",skip_header=1)
        F = File_F
        
        it = np.linspace(0, len(F)-1,len(F))
        plot_F(F,it, "Fidelity",show=True)
        
    if plot_fv: 
        file_fv = f"Figs/{file}/fv/fv{n}.csv"
        File_fv = np.genfromtxt(file_fv,delimiter = ",",skip_header=1)
        u = File_fv[0]
        fvvec = File_fv[1]
    
        plot_fv(fvvec,u,F[n],f'{n}'+r'$\times$ iteration(s) '+f'N={len(u)}',show=True)

def Refine(Cvecfvvec,u,F,i,Time,file):
    uRef = RefArray(Cvecfvvec[i], u[i])
    u.append(uRef)
    
    t1 = time.perf_counter()
    Cvecfvvec_i,F_i = find_fopt(u[i+1],file)
    Time.append(time.perf_counter()-t1)
    
    Cvecfvvec.append(Cvecfvvec_i)
    F.append(float(F_i))
    

def plot_and_save(Cvecfvvec,F,u,i,file,Time=None):
    
    plot_fv(Cvecfvvec[i],u[i],f'{i}'+r'$\times$ iteration(s) '\
            +f'N={len(u[i])}, F={F[i]:.2}, Time={Time[-1]:.0f}s',i,file)

    it = np.linspace(0, len(F)-1,len(F))
    plot_F(F,it, "Fidelity",file)
    
    File_F = np.array(F)
    np.savetxt(f"Figs/{file}/Fidelity.csv", File_F,delimiter=",",header="Fidelity:")
    
    File_fv = np.array([u[i],Cvecfvvec[i]])
    np.savetxt(f"Figs/{file}/fv/fv{i}.csv", File_fv,delimiter=",",header="Filter(u,fv):")
    
def info(file_name,points,iterations,tol=None,u0=None,pars=None):
    newpath = f"{file_name}" 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    if type(u0) != type(None):
        newpath = f"{file_name}/fv" 
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        np.savetxt(f"{file_name}/u0.csv", u0,delimiter=",",header="starting grid")
    
    with open(f"{file_name}/info.txt","w") as file:
        file.write("\n"+str(datetime.datetime.now())[:-7])
        file.write(f"\nNumber of starting points: {points}")
        file.write(f"\nNumber of iterations: {iterations}")
        file.write(f"\nMethod: {inte.method}")
        file.write(f"\nAtol={inte.Atol}, Rtol={inte.Rtol}, tpz_N={inte.N}")
        file.write("\n"+str(mp))
        if iterations == "repeat":
            file.write(f"\nTolerance: {tol}")
            
        file.write("\n\nExp. Parameters:")
        for key in p:
            file.write(f"\n{key} = {p[key]}")
            
        
        file.write("\n\nGamma_1(t) = "+str(G))
        
        if type(pars) != type(None):
            file.write(f"\nr1 = {pars}")
        
        else:
            file.write("\n\nGamma_1 Parameters:")
            for key in G.__dict__:
                file.write(f"\n{key} = {G.__dict__[key]}")

#----------------------------------------------------------------------------------------
def main_func(File_name,points,iterations,tol = 0.01):
    
    """
    Finds the optimal filter function via an iterative method and the Fidelity.
    It outputs a csv file of all the iteration of the functions and the correponding grid.
    
    Also it outputs a plot of each iteration and a plot of the fidelity as a function
    of the number of iterations

    Parameters
    ----------
    File_number : int
    
        The file number it should save to.
        
    points : int
    
        The number of starting points. 
        
    iterations : int or str
    
        Number of iterations to perfrom
        If set to 'repeat' then repeat until the specified tolerance is reachead,
        see `tol` below.

    tol : float, optional
            
        Abosulute tolerance. The default is 0.001.
    
        The specified tolerance for when 'repeat' should stop.
        Defined as the absolute difference between the last two calculated Fidelities
        ``abs(F[i+1])-F[i]<=tol``
        
        
        
        

    Returns
    -------
    None.

    """
    
    clear_cache()
        
    if type(File_name) != str or type(points) != int or\
       ( type(iterations) != int and type(iterations) != str)\
        or (type(iterations)==str and iterations != "repeat")\
            or type(tol) != float:
        raise ValueError("Incorrect type or value")
     
    
    file,N = File_name,points
    F = []
    u = []
    Cvecfvvec = []
    Time = []
    
    u0 = np.array([])
    u0 = np.linspace(0, 1, num=int((N+1)), endpoint=True)    
    u0 = mp.matrix(np.sort(np.unique(u0)))
    
    info(f"{File_name}",len(u0),iterations,tol,u0)
 
    t1 = time.perf_counter()
    Cvecfvvec0, F0 = find_fopt(u0,file,u)
    Time.append(time.perf_counter()-t1)
    
    u.append(u0)
    Cvecfvvec.append(Cvecfvvec0)
    F.append(float(F0))
    
    plot_and_save(Cvecfvvec, F, u, 0, file,Time)
        
    if iterations != "repeat":
        n = iterations
            
        for i in range(0,n):
            Refine(Cvecfvvec, u, F, i,Time,file)
            plot_and_save(Cvecfvvec, F, u, i+1, file,Time=Time)
    
    else:
        uRef = RefArray(Cvecfvvec[0], u[0])
        u.append(uRef)
        Cvecfvvec_1,F_1 = find_fopt(u[1])
        Cvecfvvec.append( Cvecfvvec_1)
        F.append(F_1)
        plot_and_save(Cvecfvvec, F, u, 0, file)
        plot_and_save(Cvecfvvec, F, u, 1, file)
        
        i = 0 
        while abs(F[i+1]-F[i])>tol:
            i+=1
            Refine(Cvecfvvec, u, F, i)
            plot_and_save(Cvecfvvec, F, u, i+1,file)
 
def noise(parameters,points,iterations,method=None,file=None,r=None,sr=None):
    clear_cache()
  
    if not (type(parameters) == type([]) or
        type(parameters) == type(np.array([]))):
        parameters = [prec(parameters)]
 
    dic = G.__dict__

    for i,key in enumerate(dic):
        setattr(G,key,prec(parameters[i]))

    N = points
    n = iterations
    Noise = []
    u = []
    Cvecfvvec = []
    
    u0 = np.linspace(0, 1, num=N+1, endpoint=True)

    Noise0,_,fvvec0 = find_fopt_sub(u0,uarr=u)
  
    u.append(u0)
    Cvecfvvec.append(fvvec0)
    Noise.append(Noise0)
    
    for i in (range(0,n)):
        uRef = RefArray(Cvecfvvec[i], u[i])
        u.append(uRef)
        noise_i,_,fvec_i = find_fopt_sub(u[i+1])
        Cvecfvvec.append(fvec_i)
        Noise.append(noise_i)
    
    if file != None:
        
        with open(f"{file}/total_output_{r}_{sr}.txt","a") as file:
            file.write("\n"+str(method)+":")
            for key in G.__dict__:
                file.write(f"\n{key} = {G.__dict__[key]}")
            file.write(f"\nN = {Noise[-1]}\n")

    return float(Noise[-1])                  
        
def OptimizeMain(): 
    global G
    G = Gamma_exp2()
    r1_lb,r1_ub = -20,0# r1 lower and upper bound
    r2_lb,r2_ub = 0,20# r2 lower and upper bound

    bounds = ((r1_lb,r1_ub),(r2_lb,r2_ub))
    par = len(bounds)
    #x0 = [rand(b[0]/2,b[1]/2) for b in bounds]
    
    #p["zeta1"] = -0.3
    methods = ['SLSQP']#['SLSQP','COBYLA']#['Nelder-Mead','L-BFGS-B','TNC','Powell']
    
    runs = 5 #Number of runs per method.
    sub_runs =  1 #Number of sub runs per run.
    
    
   
    for n in range(runs):
        N = 60#random.randint(200,250)
        ite = 0#3+n#random.randint(10,20)
        x0 = [rand(b[0],b[1]) for b in bounds]
        #print(x0)
        for method in methods:
            
            if method == 'SLSQP':
                options = {"maxiter":30,"eps":1e-5}
                
            else:
                options = {"maxiter":30}
            
            for m in range(sub_runs):
                File = f"optimize/{method}/opt_{method}_{par}par_2"
                
                if not os.path.exists(File):
                    os.makedirs(File)
                    
                p["zeta1"] = -0.3+0.1*m
                
               
                t1 = time.perf_counter()
                try:
                    result = opt.minimize(noise,x0,
                                      args=(N,ite,method,File,n,m),
                                      method=method,
                                      bounds=bounds,                                      
                                      tol=1e-4,
                                      options=options)
                                      #options={"ftol":1e-6,"gtol":1e-6,"disp":None})
                                      
                except Exception as exc:
                    
                    with open(f"{File}/total_output.txt","a") as file:
                        file.write("\n"+str(method)+":")
                        for key in G.__dict__:
                            file.write(f"\n{key} = {G.__dict__[key]}")
                        file.write(f"\n{str(exc)}")
                    
                
                t2 = time.perf_counter()
                
                with open(f"{File}/run{n}_sub{m}_0.txt","w") as file:
                   file.write(str(datetime.datetime.now())[:-7])
                   file.write("\n"+str(result)+"\n")
                   file.write(f"\nx0 = {x0}")
                   file.write(f"\nBounds = {bounds}")
                   file.write(f"\nNumber of starting point: {N}")
                   file.write(f"\nNumber of iterations: {ite}")
                   file.write(f"\nTime: {t2-t1:.3f}s")
                   
                   file.write("\n\nExp. Parameters:")
                   for key in p:
                       file.write(f"\n{key} = {p[key]}")

                   file.write("\n\nGamma_M(t) = "+str(G))
                   
                   file.write("\n\nGamma_M Parameters:")
                   for key in G.__dict__:
                       file.write(f"\n{key} = {G.__dict__[key]}")

def NoiseMain(file,pars):     

    
    info(f"{file}",N,it,pars)

    Noise = [noise(a,N,it) for a in pars]
    File_F = np.array([pars,Noise])
    np.savetxt(f"{file}/Noise.csv", File_F,delimiter=",",header="a,Noise:")
    
    plt.figure()
    plt.title(f"noise as a function of $a$, N={len(pars)}")
    plt.plot(pars,Noise,"bo",label = r"$Noise$")
    plt.plot(pars,Noise,"b-")
    plt.legend()
    plt.grid()
    plt.savefig(f"{file}/Noise")
    plt.close()

        
def FilterMain(file,pars):
    
    clear_cache()
    t1 = time.perf_counter()
    main_func(file, N, it)
    Dt = time.perf_counter()-t1
    with open(f"{file}/info.txt","a") as file:
       file.write(f"\nTime: {Dt:.2f}")
    plt.close("all")

N = 60
it = 0
mp.dps = 20

if  __name__ == "__main__":
    global G
    G = Gamma_exp2
    pars = []
    file = ""
    #NoiseMain(file)
    #OptimizeMain(file)
    FilterMain(file,pars)

