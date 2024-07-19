import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs

from numpy.linalg import inv


# la densité non tronquée et la densité tronquée sur [a,b]

def p(m,s,x): 
    y = (x-m)/s
    return  1/s * np.exp( -y -np.exp(-y))

# cdf de p

def F(x):
    return np.exp(-np.exp(-x))

def p_prime(x):
    return (-1+np.exp(-x))*p(0,1,x)

# This is the Gumbel density truncated on [a,b] and renormalized

def q(m,s,x,a,b): 
    y = (x-m)/s
    if x>b or x<a:
        return 0
    else:
        y_b = (b-m)/s
        y_a = (a-m)/s
        P_theta = F(y_b) - F(y_a)  # this is the normalization constant
        return p(m,s,x)/P_theta

def N(m,s,a,b):
    y_b = (b-m)/s
    y_a = (a-m)/s
    return F(y_b) - F(y_a) 
    
    
def grad_Hess_logN(m,s,a,b):
    grad_logN=np.zeros(2)
    Hess_logN=np.zeros((2,2))
    
    b_theta = (b-m)/s
    a_theta = (a-m)/s
    
    partial_m_a_theta, partial_m_b_theta = -1/s,-1/s
    partial_s_a_theta, partial_s_b_theta = -a_theta/s, -b_theta/s    
    partial_mm_a_theta, partial_mm_b_theta = 0, 0
    partial_ms_a_theta, partial_ms_b_theta = 1/s**2, 1/s**2
    partial_ss_a_theta, partial_ss_b_theta = 2*a_theta/s**2, 2*b_theta/s**2
    
    Nmsab = N(m,s,a,b)
    
    pmsb = p(m,s,b)
    pmsa = p(m,s,a)
    p01b_theta = p(0,1,b_theta)
    p01a_theta = p(0,1,a_theta)
    
    #grad logN
    grad_logN[0] = (-pmsb + pmsa)/Nmsab
    grad_logN[1] = (-pmsb*b_theta + pmsa*a_theta)/Nmsab
    
    #Hess logN
        #Hess12
    A1ms = p_prime(b_theta)* partial_m_b_theta * partial_s_b_theta + p01b_theta*partial_ms_b_theta 
    A2ms = p_prime(a_theta)* partial_m_a_theta * partial_s_a_theta + p01a_theta*partial_ms_a_theta
    
    Bms = p01b_theta*partial_s_b_theta - p01a_theta*partial_s_a_theta
    
    partial_m_N = -pmsb + pmsa
    
    Hess_logN[0,1] = ((A1ms-A2ms)*Nmsab - partial_m_N*Bms)/Nmsab**2
    Hess_logN[1,0] = Hess_logN[0,1]
    
        #Hess11
    A1mm = p_prime(b_theta)* partial_m_b_theta**2 + p01b_theta*partial_mm_b_theta 
    A2mm = p_prime(a_theta)* partial_m_a_theta**2 + p01a_theta*partial_mm_a_theta
    
    Bmm = p01b_theta*partial_m_b_theta - p01a_theta*partial_m_a_theta
    
    partial_m_N = p01b_theta*(-1/s) - p01a_theta*(-1/s)
    
    Hess_logN[0,0] = ((A1mm-A2mm)*Nmsab - partial_m_N*Bmm)/Nmsab**2
        
        #Hess22
    A1ss = p_prime(b_theta)* partial_s_b_theta**2 + p01b_theta*partial_ss_b_theta 
    A2ss = p_prime(a_theta)* partial_s_a_theta**2 + p01a_theta*partial_ss_a_theta
    
    Bss = p01b_theta*partial_s_b_theta - p01a_theta*partial_s_a_theta
    
    partial_s_N = p01b_theta*(-b_theta)/s - p01a_theta*(-a_theta/s)
    
    Hess_logN[1,1] = ((A1ss-A2ss)*Nmsab - partial_s_N*Bss)/Nmsab**2
    
    return grad_logN,Hess_logN
    
    
# non truncated fisher information

# Cas non tronquée
gam = 0.57721566

def func2(x):
    return np.log(x)**2 *x* np.exp(-x)

def fun3(x):
    return -np.log(x)*np.exp(-x)*x

def fun4(x):
    return np.exp(-2*x-np.exp(-x))*x**2


gam_der = integrate.quad(func2,0.001,1000)
gam_der_2 = integrate.quad(fun3,0.0001,1000)
gam_der_3 = integrate.quad(fun4,-100,100)
print(gam_der[0])
print(gam_der_2[0])
print(gam_der_3[0])
def I(m,s):
    I_11 = 1/s**2
    I_12 = (gam-1)/s**2
    I_22 = (gam_der[0]+1)/s**2
    
    return np.array([[I_11,I_12],
                     [I_12,I_22]])

# Cas tronquée

def fun(y):
    return y*p(0,1,y)
    

    
def fun2(y):
    return y**2*np.exp(-2*y - np.exp(-y))

def fun1(y):
    return y*p(0,1,y)
    

def J(m,s,a,b):
    a_theta = (a-m)/s
    b_theta = (b-m)/s
    
    grad,Hess = grad_Hess_logN(m,s,a,b)
    Nmsab = N(m,s,a,b)
    
    
    EX = integrate.quad(fun, a_theta,b_theta)[0]/Nmsab
    EeX = 1 - s*grad[0]
    EXeX  = -1 - s*grad[1] + EX 
    EX2eX = integrate.quad(fun2,a_theta,b_theta)[0]/Nmsab 

    J_11 = EeX/s**2 + Hess[0,0] # vérifié
    J_12 = 1/s**2 - EeX/s**2 + EXeX/s**2 + Hess[0,1]  # vérifié
    J_22 = 1/s**2 * ( 1 + 2*s*grad[1] + EX2eX) + Hess[1,1] # vérifié

    return np.array([[J_11,J_12],
                     [J_12, J_22]])


# partial derivatives of the fisher metric

def partial_J(typ,m,s,a,b):
    eps = 0.000000001

    if typ == "m":
        return (J(m + eps,s,a,b) - J(m,s,a,b))/(eps)
    
    if typ == "s":
        return (J(m,s+eps,a,b) - J(m,s,a,b))/(eps)
    
m,s,a,b = 0,1,-1,1

print(partial_J("m",m,s,a,b)[0,0] )

# Christoffel symbols

def ind(i):
    if i==1:
        return "m"
    if i==2:
        return "s"
    
def J_inv(m,s,a,b):
    J_inverse = np.linalg.inv(J(m,s,a,b))
    return J_inverse

def coef(i,s):
    if i==1:
        return 0
    else:
        return -2/s**3

A = np.array([[1,gam -1],[gam -1, gam_der[0]+1]])
det_A = np.linalg.det(A)
B = np.array([[gam_der[0]+1, 1-gam],[1-gam, 1]])  # transposée de la comatrice de A

def Gamma_Gum_non_tr(i,j,k,m,s):
    L = [ B[k-1,l-1]*s**2/det_A*(  coef(j,s)*A[l-1,i-1] +coef(i,s)*A[l-1,k-1] -coef(l,s)*A[i-1,j-1] ) for l in [1,2]]
    return np.sum(L)

def Gamma_Gum(i,j,k,m,s,a,b):
    J_inverse = J_inv(m,s,a,b)
    L = [0.5*J_inverse[k-1,l-1]*(partial_J(ind(j),m,s,a,b)[l-1,i-1] + partial_J(ind(i),m,s,a,b)[l-1,k-1] - partial_J(ind(l),m,s,a,b)[i-1,j-1]) for l in [1,2]]
    return np.sum(L)
    
    
# champ de vecteurs pour résoudre géodésiques

# Champ de vecteurs

def H_Gum(m,s,dm,ds,a,b):
    
    A = -Gamma_Gum(1,1,1,m,s,a,b)*dm**2 -2*Gamma_Gum(1,2,1,m,s,a,b)*dm*ds -Gamma_Gum(2,2,1,m,s,a,b)*ds**2
    B = -Gamma_Gum(1,1,2,m,s,a,b)*dm**2 -2*Gamma_Gum(1,2,2,m,s,a,b)*dm*ds -Gamma_Gum(2,2,2,m,s,a,b)*ds**2

    return np.array([dm,ds,A,B])

def H_Gum_non_tr(m,s,dm,ds):
    A = -Gamma_Gum_non_tr(1,1,1,m,s)*dm**2 -2*Gamma_Gum_non_tr(1,2,1,m,s)*dm*ds -Gamma_Gum_non_tr(2,2,1,m,s)*ds**2
    B = -Gamma_Gum_non_tr(1,1,2,m,s)*dm**2 -2*Gamma_Gum_non_tr(1,2,2,m,s)*dm*ds -Gamma_Gum_non_tr(2,2,2,m,s)*ds**2

    return np.array([dm,ds,A,B])

# geodesic solver

def geod_Gumb_tronquees(Tf,h,Y_0,a,b):
    
    N = np.int64(Tf/h)
    Y=np.zeros((N,4))
    Y[0,:] = Y_0
    
    for n in range(N-1):
        Y[n+1,:]= Y[n,:] + h*H_Gum(Y[n,0],Y[n,1],Y[n,2],Y[n,3],a,b)
        
    return Y[:,0],Y[:,1]

def geod_Gumb_non_tronquees(Tf,h,Y_0):
    
    N = np.int64(Tf/h)
    Y=np.zeros((N,4))
    Y[0,:] = Y_0
    
    for n in range(N-1):
        Y[n+1,:]= Y[n,:] + h*H_Gum_non_tr(Y[n,0],Y[n,1],Y[n,2],Y[n,3])
        
    return Y[:,0],Y[:,1]
    
# tracer géodésiques et sphères pour gumbel tronquées


def tracer_sphere_tr_avec_geod_Gumb(p,delta,nb_pts,h,a,b):
    m=p[0]
    s=p[1]
  
    L = np.linspace(0,2*np.pi,nb_pts)
    j = 0    
    n = np.int64(Tf/h)
    C = np.zeros((nb_pts,2))
    Liste = []
    for t in L:
        v = np.array([np.cos(t),np.sin(t)])
        l_J = np.sqrt(np.dot(np.transpose(v),np.dot(J(m,s,a,b),v)))
        v_J = delta*(v/l_J)
        
        X0_J = np.array([m,s,v_J[0],v_J[1]])
        
        X,Y = geod_Gumb_tronquees(1,h,X0_J,a,b)

        Liste.append([X,Y])
        
        C[j,:] = np.array([ X[-1],Y[-1] ])

        print("j=",j)
        j+=1
        

    return n,C,np.array(Liste)





