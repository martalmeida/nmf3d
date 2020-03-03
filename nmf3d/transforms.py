import numpy as np
from . import constants as const
from . import calcs
import scipy.special

dType='d'
cType=np.complex128

def vertical(u,hk,nk,Gn,p_old,p_new,dataLabel,**kargs):
  '''Vertical transform

  u, variable defined at pressure levels
  hk, equivalent heights
  nk, total number of equivalent heights
  Gn, vertical structure functions
  p_old, original pressure levels
  p_new, Gaussian pressure levels
  dataLabel, variable type:
    'zonal wind', u-wind
    'meridional wind', v-wind
    'geopotential', phi (z)
    'I1', zonal component of the kinetic energy interaction term (square brakets of eq. A16, ref1)
    'I2', meridional component of the kinetic energy interaction term (square brakets of eq. A17, ref1)
    'J3', available potential energy interaction term (square brakets of eq. A18, ref1)
  kargs:
    save, create file [False]
    format, file format: [nc] or npz
    attrs, attributes to save [{}]
    label, start of the saved filename ['out']

  Returns the vertical transform as well as saved filename if save is true

  References
    ref1:
    Castanheira, JM, Marques, CAF (2019). The energy cascade associated with
    daily variability of the North Atlantic Oscillation, Q J R Meteorol Soc.,
    145: 197– 210. doi: https://doi.org/10.1002/qj.3422
  '''

  ws0=np.isinf(hk[0])
  hk=hk[:nk]
  if ws0:
    hk=hk.copy()
    hk[0]=1.

  GL=Gn.shape[1]
  nTimes,nz,nLat,nLon=u.shape

  Gp,Gw = scipy.special.orthogonal.p_roots(GL)
  Gp=Gp[::-1]

  if not ws0 and (dataLabel=='zonal wind' or dataLabel=='meridional wind'): # U anv V only
    u[:,-1]=0 # The non-slip lower boundary condition (Tanaka and Kung, 1988)

  print (' - %s - interpolate p to sigma'%dataLabel)
  # Interpolating u from pressure leves to sigma levels (uses cubic splines)
  # Find the B-spline representation of 1-D curve. Given the set of data points (x[i], y[i])
  # determine a smooth spline approximation of degree k on the interval xb <= x <= xe.
  # The coefficients, c, and the knot points, t, are returned. Uses the FORTRAN routine curfit from FITPACK.
  us = np.zeros((nLat,nLon,nTimes,GL),dtype=dType)
  for ti in range(nTimes):
    for  la in range(nLat):
      for  lo in range(nLon):
        #Aux = splrep(P,u[ti,:,la,lo],w=None,xb=None,xe=None,k=3,task=0,s=None,t=None,full_output=0,per=0,quiet=1)
        ## u at the Gaussian sigma levels
        #us[la,lo,ti,:] = splev(p_new, Aux, der=0, ext=0);

        ## Alternatively: Is faster using function "UnivariateSpline" instead of "splrep"
        #Aux = UnivariateSpline(P, np.squeeze(u[ti,:,la,lo]), w=None, bbox=[None, None], k=3, s=0)
        ## u at the Gaussian sigma levels
        #us[la,lo,ti,:] = Aux(p_new);

        # Alternative: Doing all at once
        us[la,lo,ti,:] = scipy.interpolate.UnivariateSpline(p_old, u[ti,:,la,lo], w=None, bbox=[None, None], k=3, s=0)(p_new)

  # vertical derivative of Gn
  if dataLabel=='J3':
    Dp=np.diff(p_new)
    d_Gn_dpnew=np.zeros(Gn.shape)
    if np.isinf(hk[0]): # ws0 True
      d_Gn_dpnew[:,0]=0 # lower boundary condition
    else:
      d_Gn_dpnew[:,0]=(Gn[:,1]-Gn[:,0])/Dp[0] # forward differences

    d_Gn_dpnew[:,GL-1]=(Gn[:,-1]-Gn[:,-2])/Dp[-1] # backward differences
    for p in range(1,GL-1):
      d_Gn_dpnew[:,p]=(Dp[p-1]**2*Gn[:,p+1]-Dp[p]**2*Gn[:,p-1]-(Dp[p-1]**2-Dp[p]**2)*Gn[:,p])/(Dp[p-1]*Dp[p]*(Dp[p-1]+Dp[p]))


  print (' - %s - vertical transform'%dataLabel)
  # Vertical transform
  u_k = np.zeros((nk,nLat,nLon,nTimes),dtype=dType)
  for kk in range(nk):
    Aux = 0
    for s in range(GL):
      if dataLabel=='J3':
        Aux+=-0.5*us[:,:,:,s]*d_Gn_dpnew[kk,s]*Gw[s]
      else:
        Aux+=0.5*us[:,:,:,s]*Gn[kk,s]*Gw[s]

    if dataLabel=='geopotential':
      u_k[kk] = Aux/(const.g*hk[kk])
    elif dataLabel=='zonal wind' or dataLabel=='meridional wind':
      u_k[kk] = Aux/np.sqrt(const.g*hk[kk])
    elif dataLabel=='I1' or dataLabel=='I2':
      u_k[kk] = Aux/(2*const.Om*np.sqrt(const.g*hk[kk]))
    elif dataLabel=='J3':
      u_k[kk] = Aux/(2*const.Om)


  save  = kargs.get('save',False)
  if save:
    if   dataLabel=='geopotential':    vname='z_k';
    elif dataLabel=='zonal wind':      vname='u_k';
    elif dataLabel=='meridional wind': vname='v_k';
    elif dataLabel=='I1':              vname='I1_k';
    elif dataLabel=='I2':              vname='I2_k';
    elif dataLabel=='J3':              vname='J3_k';

    fsave=save_out({vname:u_k},**kargs);
    return u_k,fsave
  else: return u_k


def save_out(data,**kargs):
  label=kargs.get('label','out')
  format=kargs.get('format','nc') # nc or npz
  attrs=kargs.get('attrs',{})

  import platform
  import sys
  import scipy
  attrs['platform']=platform.platform()
  attrs['environment']='python'
  attrs['version']=sys.version.replace('\n','')
  attrs['version_scipy']=scipy.__version__
  attrs['version_numpy']=np.__version__

  if   'z_k'  in data: label2='z';
  elif 'u_k'  in data: label2='u';
  elif 'v_k'  in data: label2='v';
  elif 'I1_k' in data: label2='I1';
  elif 'I2_k' in data: label2='I2';
  elif 'J3_k' in data: label2='J3';

  fsave='%s_%s_k.%s'%(label,label2,format)

  print('saving %s'%fsave)
  if format=='npz':
    data.update(attrs)
    np.savez(fsave, **data)
  elif format=='nc':
    save_nc(fsave,data,**attrs)
  else: print('Unknown format, use nc or npz')

  return fsave


def save_nc(fname,data,**attrs):
  import netCDF4
  import os
  if os.path.isfile(fname):
    os.unlink(fname)

  nc=netCDF4.Dataset(fname,'w',file_format='NETCDF4_CLASSIC')

  if   'z_k'  in data:
    vname='z_k'
    vlname='Vertical transform of geopotential'
  elif 'u_k'  in data:
    vname='u_k'
    vlname='Vertical transform of zonal wind'
  elif 'v_k'  in data:
    vname='v_k'
    vlname='Vertical transform of meridonal wind'
  elif 'I1_k' in data:
    vname='I1_k'
    vlname='Vertical transform of term I1'
  elif 'I2_k' in data:
    vname='I2_k'
    vlname='Vertical transform of term I2'
  elif 'J3_k' in data:
    vname='J3_k'
    vlname='Vertical transform of term J3'

  v=data[vname]

  # dimensions:
  nk,nLat,nLon,nTime=v.shape
  nc.createDimension('number_equivalent_heights',nk)
  nc.createDimension('latitude',nLat)
  nc.createDimension('longitude',nLon)
  nc.createDimension('time',nTime)

  # variables:
  dim='number_equivalent_heights','latitude','longitude','time'
  vnc=nc.createVariable(vname,dType,dim)
  vnc.long_name=vlname

  # global attributes:
  import datetime
  nc.date=datetime.datetime.now().isoformat(' ')
  for k in attrs.keys():
    setattr(nc,k,attrs[k])

  # fill variables
  nc.variables[vname][:]=v
  nc.close()


def hough(HOUGHs_UVZ,W_nk,Lat,ws0):
  '''Hough transform

  HOUGHs_UVZ, Hough functions
  W_nk, vertical and Fourier transforms of (u,v,z) or (I1,J2) or (J3) (see transform.vertical)
  Lat, latitudes of the data and Houghs
  ws0, if true the pressure vertical velocity is zero at surface
  '''

  # check if linear or gaussian
  if np.unique(np.diff(Lat)).size==1: # linear
    Dl  = (Lat[1]-Lat[0])*np.pi/180 # Latitude spacing (radians)
    latType='linear'
  else: # gaussian
    gw=np.polynomial.legendre.leggauss(Lat.size)[1]
    latType='gaussian'

  # Eliminate geopotential/J3 (used only for reconstruction) if ws0:
  if ws0:
    HOUGHs_UVZ[2,:,:,0]=0.

  # dimensions:
  three,nM,nL,nk,nLat=HOUGHs_UVZ.shape
  nTimes=W_nk.shape[4]

  print (' - computing')
  #cosTheta = np.cos(Lat*np.pi/180)
  cosTheta = calcs.cosd(Lat)
  w_nlk = np.zeros((nk,nM,nL,nTimes), dtype=cType)
  for k in range(nk): # vertical index
    for n in range(nM): # wavenumber index
      for l in range(nL): # meridional index
        for t in range(nTimes): # time index
          Aux = W_nk[:,k,:,n,t]*np.conjugate(HOUGHs_UVZ[:,n,l,k,:])*cosTheta   # Aux(3,Lat)
          y1=Aux.sum(0) # Integrand -> y1(Lat)

          if latType=='linear':
            # Computes latitude integral of the Integrand using trapezoidal method
            aux1=(y1[:-1]+y1[1:])*Dl/2. # len Lat.size-1; complex

          elif latType=='gaussian':
            aux1=gw*y1

          w_nlk[k,n,l,t] = aux1.sum()

  return w_nlk
