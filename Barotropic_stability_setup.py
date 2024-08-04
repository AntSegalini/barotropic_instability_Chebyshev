import numpy as np

from settings import parameters,Zonal_velocity
from create_operators import create_operators,ZETA_function
import pickle

##################################################################################################

def calculate_eigenfunctions(m,theta,D1,D2,lambdax,U,DER_ZETA_abs):
    N=theta.shape[0]-1
    ST , CT = np.sin(theta) , np.cos(theta)
    ST[[0,-1]]=1            # a dummy value to avoid a singular warning
    COT=CT/ST

    # laplacian operator (at least the first and last rows will be replaced by the BC)
    Lapl_op= D2 + (COT*D1.T).T - np.diag((m/ST)**2)        

    A=((1j*m*U/ST+lambdax) * Lapl_op.T).T - np.diag(1j*m/ST*DER_ZETA_abs)
    B=1j*Lapl_op

    # boundary conditions
    if np.abs(m)>1:
        # both the von Neumann and Dirichlet conditions are imposed 
        # (the eigenvectors will have 0 derivative and 0 value as first and last points)
        e=D1[0,1]*D1[-1,-2]-D1[-1,1]*D1[0,-2]
        b0j=-(D1[-1,-2]*D1[0,2:-2]-D1[0,-2]*D1[-1,2:-2])/e
        bNj=-(-D1[-1,1]*D1[0,2:-2]+D1[0,1]*D1[-1,2:-2])/e
        A=A[2:-2,2:-2]+np.repeat((A[2:-2,1])[:,np.newaxis],N-3,axis=1)*b0j+np.repeat((A[2:-2,-2])[:,np.newaxis],N-3,axis=1)*bNj
        B=B[2:-2,2:-2]+np.repeat((B[2:-2,1])[:,np.newaxis],N-3,axis=1)*b0j+np.repeat((B[2:-2,-2])[:,np.newaxis],N-3,axis=1)*bNj
        omega_eig,P_hat=np.linalg.eig(np.linalg.solve(B,A))

    elif m==0:
        # only the von Neumann condition is imposed (the eigenvectors will have 0 derivative as first and last points)
        # plus Psi is zero at the north pole
        e=D1[0,1]*D1[-1,-1]-D1[-1,1]*D1[0,-1]
        b0j=-(D1[-1,-1]*D1[0,2:-1]-D1[0,-1]*D1[-1,2:-1])/e
        bNj=-(-D1[-1,1]*D1[0,2:-1]+D1[0,1]*D1[-1,2:-1])/e
        A=A[2:-1,2:-1]+np.repeat((A[2:-1,1])[:,np.newaxis],N-2,axis=1)*b0j+np.repeat((A[2:-1,-1])[:,np.newaxis],N-2,axis=1)*bNj
        B=B[2:-1,2:-1]+np.repeat((B[2:-1,1])[:,np.newaxis],N-2,axis=1)*b0j+np.repeat((B[2:-1,-1])[:,np.newaxis],N-2,axis=1)*bNj
        
        # The solution is an identity matrix with eigenvalue omega=lambdax
        omega_eig=-1j*lambdax*np.ones(N-2)
        P_hat=np.eye(N-2)
        
    else:
        # only the dirichlet condition is imposed (the eigenvectors will have 0 as first and last values)
        A=A[1:-1,1:-1]
        B=B[1:-1,1:-1]
        b0j,bNj=np.zeros(1),np.zeros(1)
        omega_eig,P_hat=np.linalg.eig(np.linalg.solve(B,A))
    
    return omega_eig, P_hat, np.linalg.inv(P_hat), np.linalg.inv(A), b0j, bNj

##################################################################################################

Omega,R_planet,lambdax,N_latitude,N_longitude,fileName,_,_=parameters()

Us=1

##################################################################################################

theta,longitude,D1,D2,m_all=create_operators(N_latitude,N_longitude)
ST , CT = np.sin(theta) , np.cos(theta)
# ST[[0,-1]]=1            # a dummy value to avoid a singular warning

#######################################################################

U=Zonal_velocity(theta)/Us
lambdax/=Us/R_planet           # attenuation
Ro=Us/(2*Omega*R_planet)       # Rossby number

f=CT/Ro

##################################################

ZETA,ZETA_abs,DER_ZETA_abs,PSI0=ZETA_function(theta,U,D1,D2,f)

saved_stuff={'U':U,'Zeta':ZETA,'PSI':PSI0,'R_planet':R_planet,'Ro':Ro,'longitude':longitude,'theta':theta,\
             'ST':ST,'CT':CT,'D1':D1,'D2':D2,'m':m_all,\
             'eigenvalues':[],'eigenvectors':[],'inverse_eigenvectors':[],\
             'inverse_A':[],'b0j':[],'bNj':[],'lambdax':lambdax}

##################################################

for m in m_all:
    Lambda,P_hat,P_hat_inv,A_inv,b0j,bNj=calculate_eigenfunctions(m,theta,D1,D2,lambdax,U,DER_ZETA_abs)
    if np.max(np.imag(Lambda))>0: print(m,np.max(np.imag(Lambda)))
    saved_stuff['eigenvalues'].append(Lambda)
    saved_stuff['eigenvectors'].append(P_hat)
    saved_stuff['inverse_eigenvectors'].append(P_hat_inv)
    saved_stuff['inverse_A'].append(A_inv)
    saved_stuff['b0j'].append(b0j)
    saved_stuff['bNj'].append(bNj)

with open(fileName+'.vjm', 'wb') as f:
    pickle.dump(saved_stuff,f, pickle.HIGHEST_PROTOCOL) 