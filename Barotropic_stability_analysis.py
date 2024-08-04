import numpy as np
import pickle
from netCDF4 import Dataset
from settings import parameters, forcing,forcing2

##################################################

def get_givenT(T,F_hat,pippo):

    m=pippo['m']
    ST,CT=pippo['ST'],pippo['CT']
    ST[[0,-1]]=1
    Ntheta,Nm=len(pippo['theta']),len(m)
    
    Streamfunction_hat=np.zeros((Ntheta,Nm),dtype=np.complex128)
    Zeta_hat=np.zeros((Ntheta,Nm),dtype=np.complex128)
    U_theta_hat=np.zeros((Ntheta,Nm),dtype=np.complex128)
    
    for i in range(Nm):

        MA=np.eye(pippo['eigenvalues'][i].shape[0])+0j
        if T!=np.inf:
            MA-=np.matmul(np.matmul(pippo['eigenvectors'][i],np.diag(np.exp(-1j*pippo['eigenvalues'][i]*T))),pippo['inverse_eigenvectors'][i])
            
        if m[i]>1:
            Streamfunction_hat[2:-2,i]=np.matmul(MA,np.matmul(pippo['inverse_A'][i],F_hat[2:-2,i]))
            Streamfunction_hat[1,i]=np.dot(pippo['b0j'][i],Streamfunction_hat[2:-2,i])
            Streamfunction_hat[-2,i]=np.dot(pippo['bNj'][i],Streamfunction_hat[2:-2,i])

        elif np.abs(m[i])==1:
            Streamfunction_hat[1:-1,i]=np.matmul(MA,np.matmul(pippo['inverse_A'][i],F_hat[1:-1,i]))
                
        else: #m=0
            Streamfunction_hat[2:-1,i]=np.matmul(MA,np.matmul(pippo['inverse_A'][i],F_hat[2:-1,i]))
            Streamfunction_hat[1,i]=np.dot(pippo['b0j'][i],Streamfunction_hat[2:-1,i])
            Streamfunction_hat[-1,i]=np.dot(pippo['bNj'][i],Streamfunction_hat[2:-1,i])

        Zeta_hat[:,i]=pippo['D2']@Streamfunction_hat[:,i]+CT/ST*(pippo['D1']@Streamfunction_hat[:,i])-(m[i]/ST)**2*Streamfunction_hat[:,i]
        # Zeta_hat[[0,-1],i]=(2-m[i]**2/2)*pippo['D2'][[0,-1]]@Streamfunction_hat[:,i]

    Zeta_hat[[0,1,-1],0]=0   
    ##################################################    

    Us=1
    R_planet=pippo['R_planet']

    Psi=Us*R_planet*np.fft.irfft(Streamfunction_hat,axis=1)         # this is the streamfunction (m2/s)
    
    U_lambda_hat=np.matmul(pippo['D1'],Streamfunction_hat)
    U_theta_hat=-((1j*m*Streamfunction_hat).T/ST).T
    # U_theta_hat[[0,-1],1]=-1j*np.matmul(pippo['D1'][[0,-1]],Streamfunction_hat[:,1])
    U_theta_hat[[0,-1]]=0
    Zeta_hat[[0,-1]]=0
    
    U_lambda=Us*np.fft.irfft(U_lambda_hat,axis=1)    # this is the zonal velocity (m/s)
    U_theta=Us*np.fft.irfft(U_theta_hat,axis=1)                     # this is the meridional velocity (m/s) (the minus converts the orientation of the positive u_\theta)
    zeta=Us/R_planet*np.fft.irfft(Zeta_hat,axis=1)                  # relative vorticity [1/s]

    return U_lambda,U_theta,zeta,Psi

##################################################

_,_,_,_,_,fileName_eigenfunctions,fileName_timeseries,T_all=parameters()

##################################################

with open(fileName_eigenfunctions+'.vjm' , 'rb') as f:
    pippo=pickle.load(f)

Us=1
R_planet=pippo['R_planet']
longitude=pippo['longitude']
theta=pippo['theta']
m=pippo['m']

Ntheta,Nlong,Nm=len(theta),len(longitude),len(m)

# F_hat=np.fft.rfft(forcing(theta,longitude),axis=1)/(Us/R_planet)**2
S=forcing2(theta,longitude,pippo['U'])
F_hat=np.fft.rfft(S,axis=1)/(Us/R_planet)**2
# import matplotlib.pyplot as plt
# plt.contourf(longitude,np.pi/2-theta,S,20);plt.colorbar()
# plt.show()

################################################

if True:
    ncfile=Dataset(fileName_timeseries+'.nc',mode='w',format='NETCDF4_CLASSIC')

    lat_dim=ncfile.createDimension('lat',len(theta))
    lat=ncfile.createVariable('lat',np.float32,('lat',))
    lat.units='degrees_north'
    lat.long_name='latitude'
    lat[:]=90-np.rad2deg(theta)

    lon_dim=ncfile.createDimension('lon',len(longitude))
    lon=ncfile.createVariable('lon',np.float32,('lon',))
    lon.units='degrees_east'
    lon.long_name='longitude'
    lon[:]=np.rad2deg(longitude)

    wave_dim=ncfile.createDimension('wavenumber',Nm)
    wavenumber=ncfile.createVariable('wavenumber',np.float32,('wavenumber',))
    wavenumber.units='none'
    wavenumber.long_name='wavenumber m'
    wavenumber[:]=m

    time_dim=ncfile.createDimension('time',len(T_all))
    time=ncfile.createVariable('time',np.float32,('time',))
    time.units='hours from start'
    time.long_name='time'
    time[:]=T_all

    Ulambda=ncfile.createVariable('U',np.float64,('time','lat','lon'))
    Ulambda.units='m/s'
    Ulambda.long_name='Zonal velocity'

    Utheta=ncfile.createVariable('V',np.float64,('time','lat','lon'))
    Utheta.units='m/s'
    Utheta.long_name='Meridional velocity'

    Streamfunction=ncfile.createVariable('PSI',np.float64,('time','lat','lon'))
    Streamfunction.units='m^2/s'
    Streamfunction.long_name='Streamfunction perturbation'

    Zeta=ncfile.createVariable('ZETA',np.float64,('time','lat','lon'))
    Zeta.units='1/s'
    Zeta.long_name='vertical vorticity'

    eigenvalues_r=ncfile.createVariable('eigenvalues_real',np.float64,('lat','wavenumber'))
    eigenvalues_r.units='1/s'
    eigenvalues_r.long_name='eigenvalues (Real part)'
    eigenvalues_i=ncfile.createVariable('eigenvalues_imag',np.float64,('lat','wavenumber'))
    eigenvalues_i.units='1/s'
    eigenvalues_i.long_name='eigenvalues (Imaginary part)'
    eigenvalues_r[:]=999
    for i in range(Nm):
        if m[i]==0:
            eigenvalues_r[:-3,i]=np.real(pippo['eigenvalues'][i])*Us/R_planet
            eigenvalues_i[:-3,i]=np.imag(pippo['eigenvalues'][i])*Us/R_planet
        elif np.abs(m[i])==1:
            eigenvalues_r[:-2,i]=np.real(pippo['eigenvalues'][i])*Us/R_planet
            eigenvalues_i[:-2,i]=np.imag(pippo['eigenvalues'][i])*Us/R_planet
        else:
            eigenvalues_r[:-4,i]=np.real(pippo['eigenvalues'][i])*Us/R_planet
            eigenvalues_i[:-4,i]=np.imag(pippo['eigenvalues'][i])*Us/R_planet
        
##################################################

i=-1
for T in T_all: 
    i+=1
    Ulambda[i,:],Utheta[i,:],Zeta[i,:],Streamfunction[i,:]=get_givenT(T*3600*Us/R_planet,F_hat,pippo)   
    Ulambda[i,:]=(Ulambda[i,:].T+Us*pippo['U']).T
    Zeta[i,:]=(Zeta[i,:].T+pippo['Zeta']*Us/R_planet).T
    Streamfunction[i,:]=(Streamfunction[i,:].T+Us*R_planet*pippo['PSI']).T

ncfile.close()
