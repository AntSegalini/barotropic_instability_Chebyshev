import numpy as np

def forcing(theta,longitude): 
    """
    This is the forcing with the divergence operator similarly to Wirth (2020)
    this code is only for a steady forcing
    """
    N_longitude=len(longitude)
    
    lambda0=np.deg2rad(30)
    theta0=np.deg2rad(60)
    sigma_theta=np.deg2rad(10)  
    sigma_lambda=np.deg2rad(10)
    h0=0.3
    F=-1e-4 * 15/6371e3/sigma_lambda**2 * h0 * np.repeat( np.exp(-(theta-theta0)**2/(2*sigma_theta**2))[:,np.newaxis],N_longitude,axis=1) * ((longitude-lambda0) * np.exp(-(longitude-lambda0)**2/(2*sigma_lambda**2)))
    F[[0,-1],:]=0   
    
    return F


def forcing2(theta,longitude,U0,H=15000): 
    import sys
    sys.path.insert(0,"/home/antonio/Documents/spectral_methods/General_routines/")
    from Fourier_class import Fourier_routines
    from create_operators import get_earth_topography

    D1=Fourier_routines().Fourier_D1_matrix(len(longitude))

    R_planet=6371e3 # Earth's radius
    Omega=7.292115e-5 # Earth's rotation rate [1/s]
    h0=1000 # m
    N_longitude=len(longitude)
    
    lambda0=np.deg2rad(30)
    theta0=np.deg2rad(60)
    sigma_theta=np.deg2rad(10)  
    sigma_lambda=np.deg2rad(10)

    # h=h0 * np.repeat( np.exp(-(theta-theta0)**2/(2*sigma_theta**2))[:,np.newaxis],N_longitude,axis=1) * np.exp(-(longitude-lambda0)**2/(2*sigma_lambda**2))
    h=get_earth_topography(theta,longitude)
    ST=np.sin(theta)
    ST[[0,-1]]=1

    F=(D1@(H-h.T)/ST*U0*np.cos(theta)).T*(2*Omega/H/R_planet)
    F[[0,-1]]=0

    return F

###########################################################

def Zonal_velocity(theta):
    """
    Distribution of Zonal velocity (Uphi) at the various colatitudes theta [m/s]

    Uphi must be zero at the poles (otherwise the relative vorticity ZETA is unbounded)
    """

    Uphi=15*np.sin(theta)       # solid body rotation [m/s]

    ############################## a zonal jet ##############################
    
    theta0=np.deg2rad(45)       # colatitude center of the jet
    sigma_theta=np.deg2rad(3)   # colatitude width of the jet
    Ujet=40              # jet intensity [m/s] 
    Uphi+=Ujet*np.exp(-(theta-theta0)**2/(2*sigma_theta**2))

    # theta0=np.deg2rad(60)              # colatitude center of the jet
    # sigma_theta=np.deg2rad(5)   # colatitude width of the jet
    # Ujet=0              # jet intensity [m/s] 
    # Uphi+=Ujet*np.exp(-(theta-theta0)**2/(2*sigma_theta**2))

    #########################################################################

    # part to remove the Uphi at the poles
    Uphi-=Uphi[0]+(Uphi[-1]-Uphi[0])/(theta[-1]-theta[0])*(theta-theta[0])
    
    return Uphi

#########################################################################

def parameters():
    """
    Parameters to setup the eigenvalues calculation
    """
    Omega=7.292115e-5               # Planet rotation   [1/s]
    R_planet=6371e3                 # Planet radius     [m]
    lambdax=1/(7*24*3600)           # attenuation       [1/s]

    N_latitude=2**8                 # number discretization points in the meridional direction  
    N_longitude=2**7                # number discretization points in the zonal direction

    fileName_eigenfunctions='eigenfunctions_jet_neutral_low'   # arbitrary filename of the saved Python picked structure

    fileName_timeseries='jet_case' # filename where the NETCDF data will be stored

    # T=[0,24,48,72,96,500,np.inf] # time [hours] (if np.inf it is intended that the infinite time solution is provided)
    T=[np.inf]

    return Omega,R_planet,lambdax,N_latitude,N_longitude,fileName_eigenfunctions,fileName_timeseries,T
