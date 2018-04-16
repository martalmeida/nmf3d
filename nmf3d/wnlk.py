import numpy as np
import scipy.interpolate
import scipy.special
import netCDF4
from . import constants as const


dType='d'
cType='complex128'

def load_ERA_I_once(f,fref=None,height=False):
  nc=netCDF4.Dataset(f)
  lon=nc.variables['lon'][:].astype(dType)
  lat=nc.variables['lat'][:].astype(dType)
  P=nc.variables['lev'][:].astype(dType) # pressure levels, Pa
  for i in nc.variables:
    if nc.variables[i].ndim==4:
       print('    - loading %s'%i)
       v=nc.variables[i][:].astype(dType)
       break

  # lat -90:90
  lat=lat[::-1]
  v=v[:,:,::-1,:]

  # subtracting reference:
  if fref:
    q=np.loadtxt(fref)[0].astype(dType)
    print('    - subtracting reference')
    for i in range(P.size): v[:,i,:,:]-=q[i]

  if height:
    print('    - geopotential height --> geopotential')
    v/=const.g

  return dict(lon=lon,lat=lat,P=P,v=v)


def load_ERA_I(fu,fv,fz,fzref,height):
  res={}

  def shortname(f):
    import os
    return '....'+os.path.join(os.path.basename(os.path.dirname(f)),os.path.basename(f))

  print('loading u : %s'%shortname(fu))
  res['u']=load_ERA_I_once(fu)
  print('loading v : %s'%shortname(fv))
  res['v']=load_ERA_I_once(fv)
  print('loading z : %s'%shortname(fz))
  res['z']=load_ERA_I_once(fz,fzref,height)
  return res


def prep(data,p_new,nk,Gn,Gw,hk,nN,ws0,label=''):
  u   = data['v']
  Lat = data['lat']
  Lon = data['lon']
  P   = data['P']

  if not ws0 and label!='geopotential': # U anv V only
    u[:,-1,:,:]=0 # The non-slip lower boundary condition (Tanaka and Kung, 1988)

  nTimes=u.shape[0]
  GL=Gn.shape[1]

  print (' - %s - interpolate p to sigma'%label)
  # Interpolating u from pressure leves to sigma levels (uses cubic splines)
  # Find the B-spline representation of 1-D curve. Given the set of data points (x[i], y[i])
  # determine a smooth spline approximation of degree k on the interval xb <= x <= xe.
  # The coefficients, c, and the knot points, t, are returned. Uses the FORTRAN routine curfit from FITPACK.
  us = np.zeros((Lat.size,Lon.size,nTimes,GL),dtype=dType)
  for ti in range(nTimes):
    for  la in range(Lat.size):
      for  lo in range(Lon.size):
        #Aux = splrep(P, np.squeeze(u[ti,:,la,lo]), w=None, xb=None, xe=None, k=3, task=0, s=None, t=None, full_output=0, per=0, quiet=1);
        ## u at the Gaussian sigma levels
        #us[la,lo,ti,:] = splev(p_new, Aux, der=0, ext=0);

        ## Alternatively: Is faster using function "UnivariateSpline" instead of "splrep"  
        #Aux = UnivariateSpline(P, np.squeeze(u[ti,:,la,lo]), w=None, bbox=[None, None], k=3, s=0);
        ## u at the Gaussian sigma levels
        #us[la,lo,ti,:] = Aux(p_new);

        # Alternative: Doing all at once
        us[la,lo,ti,:] = scipy.interpolate.UnivariateSpline(P, u[ti,:,la,lo].flat, w=None, bbox=[None, None], k=3, s=0)(p_new)


  print (' - %s - vertical transform'%label)
  # Vertical transform
  u_k = np.zeros((nk,Lat.size,Lon.size,nTimes),dtype=dType)
  for kk in range(nk):
    Aux = 0
    for s in range(GL):
      Aux+=us[:,:,:,s] * Gn[kk,s] * Gw[s]

    if label=='geopotential':
      u_k[kk] = Aux / (const.g*hk[kk])   # u_k(nk,lat,lon,nTimes) -> dimensionless
    else:
      u_k[kk] = Aux / np.sqrt(const.g*hk[kk])   # u_k(nk,lat,lon,nTimes) -> dimensionless


  print (' - %s - Fourier transform'%label)
  # Fourier transform
  U_nk = np.fft.fft(u_k,n=None, axis=2)   # U_nk is the Fourier Transform of u_k along dimension 2 (i.e along longitude).
  U_nk = U_nk / Lon.size       # Scale the fft so that it is not a function of the length of input vector.

  # Retaining the first nN (see hough_functions.py) zonal wave numbers
  # U_nk has dimensions U_nk(m,Lat,nN=Lon,t)
  U_0k = U_nk[:,:,0,:].real   # Wavenumber zero, i.e. the zonal mean (real not needed in matlab version)
  U_nk = U_nk[:,:,1:nN+1,:]   # First nN wavenumbers.

  return U_0k,U_nk


def spectrum(vfile,hfile,data,**kargs):
  '''
  Compute the Vertical, Fourier and Hough transforms of zonal and
  meridional wind, and geopotential perturbation (from the reference
  geopotential), i.e. the 3-D spectrum of total energy W_nlk.

  vfile: equivalent heights and vertical structure functions
  hfile: Hough functions
  data: u,v,phi data
  kargs:
    save, create file [True]
    format, file format: [nc] or npz
    attrs, attributes to save [{}]
    label, start of the saved filename ['out']

  Returns the expansion coefficients (w_nlk) and the zonal expansion
    coefficients (w_0lk) as well and saved filename of save is True

   '''

  save  = kargs.get('save',True)   # create file

  # check number of hfiles:
  try:
    hfile,hfile_B=hfile
  except:
    pass

  print (' - loading parameters from Hough functions file:\n    %s'%hfile)
  if hfile.endswith('.npz'):
    ohfile=np.load(hfile)

    max_zonal_wave_number,quarter_number_gravitical_modes, number_equivalent_heights=ohfile['WEST_G_sy'].shape
    half_number_rossby_modes=ohfile['WEST_R_sy'].shape[1]

    LG=quarter_number_gravitical_modes*4
    LR=half_number_rossby_modes*2
    nN=max_zonal_wave_number
    nk=number_equivalent_heights

  else: # netcdf
    # loading dimensions:
    nc=netCDF4.Dataset(hfile)
    LG=nc.dimensions['quarter_number_gravitical_modes'].size*4 # number of gravity meridional modes (should be even)
    LR=nc.dimensions['half_number_rossby_modes'].size*2        # number of Rossby meridional modes (should be even)
    nN=nc.dimensions['max_zonal_wave_number'].size             # number of zonal wavenumbers
    #nk=nc.nk # may differ from number_equivalent_heights if ws0
    nk=nc.dimensions['number_equivalent_heights'].size
    nc.close()

  nL= LG+LR                                                    # number of total meridional modes

  print (' - loading vertical structure functions:\n    %s'%vfile)
  if vfile.endswith('.npz'):
    ovfile=np.load(vfile)
    ws0=eval(ovfile['ws0'][()]) # ws0 is the string True/False

    if ws0:
      nk+=1
      hk=ovfile['hk'][:nk]
      hk[0]=1.
    else:
      hk=ovfile['hk'][:nk]

    Gn=ovfile['Gn'][:nk]

  else: # netcdf
    nc=netCDF4.Dataset(vfile)
    ws0=eval(nc.ws0)

    if ws0:
      nk+=1
      hk=nc.variables['hk'][:nk]
      hk[0]=1.
    else:
      hk=nc.variables['hk'][:nk]

    Gn=nc.variables['Gn'][:nk]
    nc.close()

  GL=Gn.shape[1]

  # Gaussian levels (i.e. points (Gp)) and Gaussian weights (Gw)
  Gp,Gw = scipy.special.orthogonal.p_roots(GL)
  Gp=Gp[::-1] # flip Gp
  p_new = (Gp+1)*const.ps/2. # Pressure levels that correspond to the chosen Gaussian sigma levels


  # zonal wind, meridional wind, geopotential  -----------------------
  U_0k,U_nk=prep(data['u'],p_new,nk,Gn,Gw,hk,nN,ws0,'zonal wind')
  V_0k,V_nk=prep(data['v'],p_new,nk,Gn,Gw,hk,nN,ws0,'meridional wind')
  Z_0k,Z_nk=prep(data['z'],p_new,nk,Gn,Gw,hk,nN,ws0,'geopotential')


  # Storing vertical and Fourier transforms of u, v and PHI:
  W_nk = np.zeros((3,)+U_nk.shape,dtype=cType)
  W_nk[0]=U_nk
  W_nk[1]=V_nk
  W_nk[2]=Z_nk

  W_0k = np.zeros((3,)+U_0k.shape,dtype=dType)
  W_0k[0,:,:,:]=U_0k
  W_0k[1,:,:,:]=V_0k
  W_0k[2,:,:,:]=Z_0k


  # Hough transforms -------------------------------------------------
  print (' - loading Hough vector functions')
  if ws0: print ('   (%s)'%hfile)
  if hfile.endswith('.npz'):
    HOUGH_UVZ_b   = ohfile['HOUGH_UVZ']
    HOUGH_0_UVZ_b = ohfile['HOUGH_0_UVZ']
  else:
    nc=netCDF4.Dataset(hfile)
    HOUGH_UVZ_b=nc.variables['HOUGH_UVZ_real'][:]+1j*nc.variables['HOUGH_UVZ_imag'][:]
    # Hough vector functions for zonal wavenumber n = 0 :
    HOUGH_0_UVZ_b=nc.variables['HOUGH_0_UVZ_real'][:]+1j*nc.variables['HOUGH_0_UVZ_imag'][:]
    nc.close()

  if ws0: # read barotropic file:
    print ('   (%s)'%hfile_B)
    if  hfile_B.endswith('.npz'):
      ohfile_B=np.load(hfile_B)
      HOUGH_UVZ_B   = ohfile_B['HOUGH_UVZ']
      HOUGH_0_UVZ_B = ohfile_B['HOUGH_0_UVZ']
    else:
      ncB=netCDF4.Dataset(hfile_B)
      HOUGH_UVZ_B=ncB.variables['HOUGH_UVZ_real'][:]+1j*ncB.variables['HOUGH_UVZ_imag'][:]
      # Hough vector functions for zonal wavenumber n = 0 :
      HOUGH_0_UVZ_B=ncB.variables['HOUGH_0_UVZ'][:]
      ncB.close()

    # concatenate:
    shape  = list(HOUGH_UVZ_b.shape)
    shape0 = list(HOUGH_0_UVZ_b.shape)
    shape[3]  = nk
    shape0[2] = nk

    HOUGH_UVZ=np.zeros(shape,cType)
    HOUGH_0_UVZ=np.zeros(shape0,cType)

    HOUGH_UVZ[:,:,:,0,:]=HOUGH_UVZ_B
    HOUGH_0_UVZ[:,:,0,:]=HOUGH_0_UVZ_B

    HOUGH_UVZ[:,:,:,1:,:]=HOUGH_UVZ_b
    HOUGH_0_UVZ[:,:,1:,:]=HOUGH_0_UVZ_b
  else:
    HOUGH_UVZ=HOUGH_UVZ_b
    HOUGH_0_UVZ=HOUGH_0_UVZ_b


  Lat=data['u']['lat']
  # check if linear or gaussian
  if np.unique(np.diff(Lat)).size==1: # linear
    Dl  = (Lat[1]-Lat[0])*np.pi/180 # Latitude spacing (radians)
    latType='linear'
  else: # gaussian
    gw=np.polynomial.legendre.leggauss(Lat.size)[1]
    latType='gaussian'

  nTimes=data['u']['v'].shape[0]

  print (' - computing')
  THETA    = Lat*np.pi/180
  cosTheta = np.cos(THETA)
  w_nlk = np.zeros((nk,nN,nL,nTimes), dtype=cType)
###  aux1  = np.zeros(Lat.size-1,  dtype=cType)
  for k in range(nk): # vertical index
    for n in range(nN): # wavenumber index
      for l in range(nL): # meridional index
        for t in range(nTimes): # time index
          Aux = W_nk[:,k,:,n,t]*np.conjugate(HOUGH_UVZ[:,n,l,k,:])*cosTheta   # Aux(3,Lat)
          y1=Aux.sum(0) # Integrand -> y1(Lat)

          if latType=='linear':
            # Computes latitude integral of the Integrand using trapezoidal method
###            for la in range(Lat.size-1):
###              aux1[la] = (y1[la]+y1[la+1]) * Dl/2.;

            aux1=(y1[:-1]+y1[1:])*Dl/2.

          elif latType=='gaussian':
            aux1=gw*y1

          w_nlk[k,n,l,t] = aux1.sum()

  # for zonal wavenumber n = 0:
  w_0lk = np.zeros((nk,nL,nTimes),   dtype=cType)
  aux0  = np.zeros(Lat.size-1, dtype=cType)
  for k in range(nk): # vertical index
    for l in range(nL): # meridional index
      for t in range(nTimes): # time index
        Aux0 = W_0k[:,k,:,t]*np.conjugate(HOUGH_0_UVZ[:,l,k,:])*cosTheta  # Aux0(3,Lat)
        y0=Aux0.sum(0) # Integrand -> y0(Lat)

        # Computes latitude integral of the Integrand using trapezoidal method
####        for la in range(Lat.size-1):
####          aux0[la] = (y0[la]+y0[la+1]) * Dl/2.;

        aux1=(y0[:-1]+y0[1:])*Dl/2.

        w_0lk[k,l,t] = aux0.sum()


  if save:
    fsave=save_out(dict(w_nlk=w_nlk,w_0lk=w_0lk),**kargs)
    return w_nlk,w_0lk,fsave
  else:
    return w_nlk,w_0lk


def save_out(data,**kargs):
  label=kargs.get('label','out')
  format=kargs.get('format','nc') # nc or npz
  attrs=kargs.get('attrs',{})

  import platform
  import sys
  attrs['platform']      = platform.platform()
  attrs['environment']   = 'python'
  attrs['version']       = sys.version
  attrs['version_scipy'] = scipy.__version__
  attrs['version_numpy'] = np.__version__

  fsave='%s_wnlk.%s'%(label,format)

  print('saving %s'%fsave)
  if format=='npz':
    data.update(attrs)
    np.savez(fsave, **data)
  elif format=='nc':
    save_nc(fsave,data,attrs)
  else: print('Unknown format, use nc or npz')

  return fsave


def save_nc(fname,data,attrs):
  import netCDF4
  import os
  if os.path.isfile(fname):
    os.unlink(fname)

  nc=netCDF4.Dataset(fname,'w',file_format='NETCDF4_CLASSIC')

  # dimensions:
  nk,nN,nL,nTimes=data['w_nlk'].shape
  nc.createDimension('number_equivalent_heights',nk)
  nc.createDimension('max_zonal_wave_number',nN)
  nc.createDimension('total_meridional_modes',nL) # LG+LR
  nc.createDimension('time',nTimes)

  # variables:
  dim='number_equivalent_heights','max_zonal_wave_number','total_meridional_modes','time'
  v=nc.createVariable('w_nlk_real',dType,dim)
  v.long_name='Expansion coefficients (real)'
  v=nc.createVariable('w_nlk_imag',dType,dim)
  v.long_name='Expansion coefficients (imag)'

  dim='number_equivalent_heights','total_meridional_modes','time'
  v=nc.createVariable('w_0lk_real',dType,dim)
  v.long_name='Zonal expansion coefficients (real)'
  v=nc.createVariable('w_0lk_imag',dType,dim)
  v.long_name='Zonal expansion coefficients (imag)'

  # global attributes:
  import datetime
  nc.date=datetime.datetime.now().isoformat(' ')
  for k in attrs.keys():
    setattr(nc,k,attrs[k])

  # fill variables
  nc.variables['w_nlk_real'][:]=data['w_nlk'].real
  nc.variables['w_nlk_imag'][:]=data['w_nlk'].imag
  nc.variables['w_0lk_real'][:]=data['w_0lk'].real
  nc.variables['w_0lk_imag'][:]=data['w_0lk'].imag

  nc.close()
