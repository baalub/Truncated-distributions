import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs

from numpy.linalg import inv


# la densité non tronquée et la densité tronquée sur [a,b]

def p(m,s,x): 
    return  1/(np.sqrt(2*np.pi)*s) * np.exp( -0.5*(x-m)**2/s**2  )
    
def q(m,s,x,a,b):
    if x>b or x<a:
        return 0
    else:
        P_theta = scs.norm(m,s).cdf(b) - scs.norm(m,s).cdf(a)
        
        return p(m,s,x)/P_theta

def N(m,s,a,b):
    return scs.norm(m,s).cdf(b) - scs.norm(m,s).cdf(a)
    
    
# Maintenant on définit mu_B et sigma_B^2

def mu_B(m,s,a,b):
    return m - s*s*(q(m,s,b,a,b) - q(m,s,a,a,b))

def sigma_B_carre(m,s,a,b):
    return s**2 * ( 1 - ((b-m)*q(m,s,b,a,b) - (a-m)*q(m,s,a,a,b))  ) - (m - mu_B(m,s,a,b) )**2

def grad_q(m,s,x,a,b):
    q_val = q(m,s,x,a,b)
    qmsbab = q(m,s,b,a,b)
    qmsaab = q(m,s,a,a,b)
    
    A = (x-m)/(s**2)  + qmsbab - qmsaab
    B = 1/s  * ( (b-m)*qmsbab  - (a-m)*qmsaab )  - 1/s  +  (x-m)**2/(s**3)
    
    return np.array([q_val*A,q_val*B])

# les dérivées partielles secondes de q_theta (densité tronquée) par rapport à mu et sigma

def Hess_q(m,s,x,a,b):
    Hess = np.zeros((2,2))
    
    qmsbab = q(m,s,b,a,b)
    qmsaab = q(m,s,a,a,b)
    qmsxab = q(m,s,x,a,b)
    
    grad_q_msxab = grad_q(m,s,x,a,b)
    grad_q_msbab = grad_q(m,s,b,a,b)
    grad_q_msaab = grad_q(m,s,a,a,b)
    
    Amm = (x-m)/(s**2) + qmsbab - qmsaab
    Bmm = -1/(s**2) + grad_q_msbab[0] - grad_q_msaab[0]
    
    Ams = (x-m)/(s**2) + qmsbab - qmsaab
    Bms = -2*(x-m)/(s**3) + grad_q_msbab[1] - grad_q_msaab[1]
    
    Ass = (1/s  * ( (b-m)*qmsbab  - (a-m)*qmsaab )  - 1/s  +  (x-m)**2/(s**3) )
    Bss = (b-m)*qmsbab - (a-m)*qmsaab  
    Css = (b-m)*grad_q_msbab[1] - (a-m)*grad_q_msaab[1]
    
    Hess[0,0] = grad_q_msxab[0]*Amm + qmsxab*Bmm
    Hess[0,1] = grad_q_msxab[1]*Ams + qmsxab*Bms
    Hess[1,0] = Hess[0,1]
    Hess[1,1] = grad_q_msxab[1]*Ass + qmsxab*(-1/(s**2) * Bss + 1/s *Css + 1/(s**2)  - 3*(x-m)**2/(s**4))
    
    return Hess,qmsaab,qmsbab,grad_q_msaab,grad_q_msbab


# Les dérivées partielles premières de mu_B et sigma_B^2

def grad_hess_de_mu_sigma(m,s,a,b):

    #Calcul le gradient et hessienne de mu_B et sigma_B

    grad_mu = np.zeros(2)
    grad_sigma = np.zeros(2)
    
    Hess_mu = np.zeros((2,2))
    Hess_sigma = np.zeros((2,2))
    
    Hess_q_msaab,qmsaab,qmsbab,grad_q_msaab,grad_q_msbab = Hess_q(m,s,a,a,b)
    Hess_q_msbab = Hess_q(m,s,b,a,b)[0]
    
    mu_B_val = mu_B(m,s,a,b)
    
    #grad_mu
    grad_mu[0] = 1 - s**2 * (grad_q_msbab[0] - grad_q_msaab[0] )
    grad_mu[1] = -2*s*(qmsbab - qmsaab)  - s**2 * (grad_q_msbab[1] - grad_q_msaab[1])
    
    #grad_sigma
    Agrad_sigma0 = -qmsbab   + (b-m)*grad_q_msbab[0] + qmsaab - (a-m)*grad_q_msaab[0]
    Bgrad_sigma0 = 1 - grad_mu[0]
    grad_sigma[0] = -s**2 * Agrad_sigma0 - 2*(m - mu_B_val)*Bgrad_sigma0
    
    Agrad_sigma1 = 1 - ( (b-m)*qmsbab - (a-m)*qmsaab )
    Bgrad_sigma1 = (b-m)*grad_q_msbab[1] - (a-m)*grad_q_msaab[1]
    grad_sigma[1] = 2*s*Agrad_sigma1 - s**2*Bgrad_sigma1 + 2*(m - mu_B_val) * grad_mu[1]
    
    #Hess_mu
    Hess_mu[0,0] = -s**2 * (Hess_q_msbab[0,0]- Hess_q_msaab[0,0])
    Hess_mu[0,1] = -2*s*(grad_q_msbab[0] - grad_q_msaab[0]) -s**2 * (Hess_q_msbab[0,1] - Hess_q_msaab[0,1])
    Hess_mu[1,0] = Hess_mu[1,0]
    Hess_mu[1,1] = -2*(qmsbab - qmsaab) - 4*s*(grad_q_msbab[1] - grad_q_msaab[1])  - s**2*(Hess_q_msbab[1,1] - Hess_q_msaab[1,1])

    #Hess_sigma
    Ahess_sigma00 =  -2*grad_q_msbab[0]  + (b-m)*Hess_q_msbab[0,0] -( -2*grad_q_msaab[0]  + (a-m)*Hess_q_msaab[0,0])
    Bhess_sigma00 = (1 - grad_mu[0])**2
    Hess_sigma[0,0] = -s**2 *Ahess_sigma00 -2*Bhess_sigma00 +2*(m - mu_B_val)*Hess_mu[0,0] 


    Ahess_sigma01 =  -qmsbab + (b-m)*grad_q_msbab[0] +qmsaab - (a-m)*grad_q_msaab[0]
    Bhess_sigma01 =  -grad_q_msbab[1] + (b-m)*Hess_q_msbab[0,1] - (-grad_q_msaab[1] + (a-m)*Hess_q_msaab[0,1])
    Chess_sigma01 =  2 *grad_mu[1]*(1 - grad_mu[0])
    Dhess_sigma01 =  2 *Hess_mu[0,1]*(m - mu_B_val)
    Hess_sigma[0,1] = -2*s*Ahess_sigma01 - s**2*Bhess_sigma01 + Chess_sigma01 + Dhess_sigma01
    Hess_sigma[1,0] = Hess_sigma[0,1]
    
    Ahess_sigma11 = 1 - ((b-m)*qmsbab - (a-m)*qmsaab)
    Bhess_sigma11 = (b-m)*grad_q_msbab[1] - (a-m)*grad_q_msaab[1] 
    Chess_sigma11 = (b - m)*Hess_q_msbab[1,1] - (a - m)*Hess_q_msaab[1,1]
    Dhess_sigma11 = 2* ( grad_mu[1])**2
    Ehess_sigma11 = 2*(m - mu_B_val)* Hess_mu[1,1] 
    Hess_sigma[1,1] = 2*Ahess_sigma11 - 4*s*Bhess_sigma11 - s**2 * Chess_sigma11 + Dhess_sigma11 - Ehess_sigma11
    
    return grad_mu,grad_sigma,Hess_mu,Hess_sigma


#la matrice de Fisher dans le cas non tronqué

def I(m,s):  
    
    i = np.zeros((2,2))
    
    i[0,0] = 1/(s**2)
    i[0,1] = 0
    i[1,0] = 0 
    i[1,1] = 2/s**2
    
    return i
    
 #la matrice de Fisher dans le cas tronqué
 
def J(m,s,a,b):   
    J_B = np.zeros((2,2))
    partial_m_mu_B_val = partial_m_de_mu_B(m,s,a,b) 
    J_B[0,0] = partial_m_de_mu_B_val/(s**2)
    J_B[0,1] = partial_s_de_mu_B_val/(s**2)
    J_B[1,0] = J_B[0,1] 
    J_B[1,1] = (partial_s_de_sigma_B_carre(m,s,a,b) + 2*(mu_B(m,s,a,b) - m) *partial_s_de_mu_B(m,s,a,b) )/(s**3)
    
    return J_B
    
    
# On code ici les symboles de Christoffel de première espèce


def Gamma_LC_prem(m,s,a,b):
    Gam = np.zeros((2,2,2))
    mu_B_val = mu_B(m,s,a,b)
    
    grad_mu,grad_sigma,Hess_mu,Hess_sigma = grad_hess_de_mu_sigma(m,s,a,b)
    
    partial_m_de_mu_B_val = grad_mu[0]
    partial_s_de_mu_B_val = grad_mu[1]
    
    partial_m_de_sigma_B_carre_val = grad_sigma[0]
    partial_s_de_sigma_B_carre_val = grad_sigma[1]
    
    partial_mm_2_mu_B_val = Hess_mu[0,0]
    partial_sm_2_mu_B_val = Hess_mu[0,1]
    partial_ss_2_mu_B_val = Hess_mu[1,1]
    
    partial_mm_2_sigma_B_carre_val = Hess_sigma[0,0]
    partial_sm_2_sigma_B_carre_val = Hess_sigma[0,1]
    partial_ss_2_sigma_B_carre_val = Hess_sigma[1,1]
    
    #\Gamma^1
    Gam[0,0,0] = partial_mm_2_mu_B_val/(s**2) /2
    Gam[0,1,0] = (partial_sm_2_mu_B_val/(s**2) -2*partial_m_de_mu_B_val/(s**3) )/2
    Gam[1,0,0] = Gam[0,1,0]
    Gam[1,1,0] = (partial_ss_2_mu_B_val/(s**2) -3/(s**4) * (partial_m_de_sigma_B_carre_val + 2*partial_m_de_mu_B_val * (mu_B_val - m)) )/2
    
    #\Gamma^2
    Gam[0,0,1] = 1/(s**3) * (partial_mm_2_sigma_B_carre_val + 2*( mu_B_val - m)*partial_mm_2_mu_B_val + 2*partial_m_de_mu_B_val**2)/2
    Gam[0,1,1] = (1/(s**3) * (partial_sm_2_sigma_B_carre_val + 2*( mu_B_val - m)*partial_sm_2_mu_B_val + 2*partial_m_de_mu_B_val*partial_s_de_mu_B_val) -2*partial_s_de_mu_B_val/(s**3))/2
    Gam[1,0,1] = Gam[0,1,1]
    Gam[1,1,1] = (1/(s**3) * (partial_ss_2_sigma_B_carre_val + 2*( mu_B_val - m)*partial_ss_2_mu_B_val + 2*partial_s_de_mu_B_val**2) -3/(s**4) * (partial_s_de_sigma_B_carre_val + 2*partial_s_de_mu_B_val * (mu_B_val- m)))/2              

    return Gam
    
def Gamma_LC(m,s,a,b):
    Gam_LC = np.zeros((2,2,2))    # Gam_LC[l,i,j] = \Gamma_{ij}^l

    J_inv = np.linalg.inv(J(m,s,a,b))
    
    Gam_prem = Gamma_LC_prem(m,s,a,b)
    
    for i in range(2):
        for j in range(2):
            for l in range(2):
                Gam_LC[i,j,l] = Gam_prem[i,j,0]*J_inv[0,l] + Gam_prem[i,j,1]*J_inv[1,l] 
                
def Gamma_LC_nt(i,j,k,m,s):
    if k == 1:
        if i == 1 and j == 2 or j == 1 and i == 2:
            return -1/s
        else:
            return 0
    
    if k == 2:
        if i == 1 and j == 1:
            return 1/(2*s)
        if i == 2 and j == 2:
            return -1/s
        else: 
            return 0                
    return Gam_LC 
    
    
# Champs de vecteurs pour les géodésiques

def H(p1,p2,v1,v2,a,b):
    Gam_LC = Gamma_LC(p1,p2,a,b)
    A1 = -Gam_LC[0,0,0]*v1**2 -2*Gam_LC[0,1,0]*v1*v2 -Gam_LC[1,1,0]*v2**2
    B1 = -Gam_LC[0,0,1]*v1**2 -2*Gam_LC[0,1,1]*v1*v2 -Gam_LC[1,1,1]*v2**2
    #A = -Gamma_LC_indice(1,1,1,p1,p2,a,b)*v1**2 -2*Gamma_LC_indice(1,2,1,p1,p2,a,b)*v1*v2 -Gamma_LC_indice(2,2,1,p1,p2,a,b)*v2**2
    #B = -Gamma_LC_indice(1,1,2,p1,p2,a,b)*v1**2 -2*Gamma_LC_indice(1,2,2,p1,p2,a,b)*v1*v2 -Gamma_LC_indice(2,2,2,p1,p2,a,b)*v2**2

    return np.array([v1,v2,A1,B1])
 
def G(x,y,u,v):
    return np.array([u,v,2*u*v/y,-u**2/(2*y) + v**2/y])
    
    
def geod_non_tronquees(Tf,h,X_0): 
    
    N = np.int64(Tf/h)
    X = np.zeros((N,4))
    X[0,:] = X_0
    
    for n in range(N-1):
        X[n+1,:] = X[n,:] + h*G(X[n,0],X[n,1],X[n,2],X[n,3])
        
    return X[:,0],X[:,1]


def geod_tronquees(Tf,h,Y_0,a,b):
    
    N = np.int64(Tf/h)
    Y=np.zeros((N,4))
    Y[0,:] = Y_0
    
    for n in range(N-1):
        Y[n+1,:]= Y[n,:] + h*H(Y[n,0],Y[n,1],Y[n,2],Y[n,3],a,b)
        
    return Y[:,0],Y[:,1]
    
    
#fonction qui donne les géodésiques partant du centre vers la sphère

def tracer_sphere_tr_avec_geod(p,delta,nb_pts,h,a,b,col):
    p1=p[0]
    p2=p[1]
    #nb_pts = 
    L = np.linspace(0,2*np.pi,nb_pts)
    j = 0    
    n = np.int64(Tf/h)
    C = np.zeros((nb_pts,2))
    Liste = []
    for t in L:
        v = np.array([np.cos(t),np.sin(t)])
        l_J = np.sqrt(np.dot(np.transpose(v),np.dot(J(p1,p2,a,b),v)))
        v_J = delta*(v/l_J)
        
        X0_J = np.array([p1,p2,v_J[0],v_J[1]])
        
        X,Y = geod_tronquees(1,h,X0_J,a,b)

        Liste.append([X,Y])
        
        C[j,:] = np.array([ X[-1],Y[-1] ])
        #plt.plot(X,Y,color="blue")
        print(j)
        j+=1
        
    #plt.plot(C[:,0],C[:,1],color=col,label='radius='+str(delta))
    
    return n,C,np.array(Liste)
