import numpy as np
import sys
sys.path.insert(0,"/home/antonio/Documents/spectral_methods/General_routines/")
from Chebyshev_class import Chebyshev_routines

def create_operators(N_latitude,N_longitude):

    C=Chebyshev_routines()
    theta=(1-C.Cheb_x(N_latitude))*np.pi/2  # colatitude
    factor=-2/np.pi
    D1=C.Cheb_D1_matrix(N_latitude)*factor
    D2=C.Cheb_D2_matrix(N_latitude)*factor**2

    longitude=np.linspace(0,2*np.pi*(1-1/N_longitude),N_longitude)

    m_all=np.fft.rfftfreq(N_longitude,1/N_longitude)

    return theta,longitude,D1,D2,m_all

############################################################################

def ZETA_function(theta,U,D1,D2,f):
    Ntheta=len(theta)
    ST , CT = np.sin(theta) , np.cos(theta)

    ZETA=np.zeros(Ntheta)
    ZETA[1:-1]= CT[1:-1]/ST[1:-1]*U[1:-1] + D1[1:-1]@U
    ZETA[[0,-1]]=2*D1[[0,-1]]@U

    A=np.zeros((Ntheta,Ntheta))
    A[2:-1]=D2[2:-1]+(CT[2:-1]/ST[2:-1]*D1[2:-1].T).T
    A[[1,-1]]=D1[[0,-1]]
    A[0]=0;A[0,0]=1

    B=ZETA.copy()
    B[[0,1,-1]]=0
    PSI0=np.linalg.solve(A,B)

    ZETA_abs=ZETA+f

    DER_ZETA_abs=D1@ZETA_abs

    return ZETA,ZETA_abs,DER_ZETA_abs,PSI0

############################################################################

def get_earth_topography(theta,longitude):
    from netCDF4 import Dataset
    from scipy.interpolate import interpn
    
    my_example_nc_file = './geo_1279l4_0.1x0.1.grib2_v4_unpack.nc'
    fh = Dataset(my_example_nc_file, mode='r')
    # print(fh.variables)

    lons = fh.variables['longitude'][:]*np.pi/180
    colats = (90-fh.variables['latitude'][:])*np.pi/180
    PHI = np.squeeze(fh.variables['z'][:]) #geopotential
    fh.close()
    THETA,LONGITUDE=np.meshgrid(theta,longitude,indexing='ij')
    return interpn((colats,lons),PHI/9.8066,(THETA.flatten(),LONGITUDE.flatten())).reshape((len(theta),-1))

# theta=np.linspace(0,np.pi,200)
# longitude=np.linspace(0,2*np.pi-0.1,150)
# THETA,LONGITUDE=np.meshgrid(theta,longitude,indexing='ij')
# H=get_earth_topography(THETA.flatten(),LONGITUDE.flatten())

# import matplotlib.pyplot as plt
# plt.figure()
# plt.contourf(longitude,theta,H.reshape((len(theta),-1)));plt.colorbar()
# plt.show()