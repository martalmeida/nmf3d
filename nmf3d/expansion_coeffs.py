import numpy as np
import scipy.interpolate
import scipy.special
import netCDF4
from . import transforms

dType='d'
cType='complex128'

def calc(vfile,hfile,data,**kargs):
  '''
  Expansion coefficients
  Compute the Vertical, Fourier and Hough transforms of:
    - zonal wind, meridional wind and geopotential perturbation (from the
      reference  geopotential), used for the the 3-D spectrum of total energy
      W_nlk
    - I1, I2 and J3 (see transforms.vertical), used for the 3-D spectrum of
      energy interactions (kinetic and available pontential energy)

  vfile: equivalent heights and vertical structure functions
  hfile: Hough functions
  data: dict with fields (u,v,z) or (I1,I2) or (J3)
  kargs:
    save, create file [True]
    format, file format: [nc] or npz
    attrs, attributes to save [{}]
    label, start of the saved filename ['out']

  Returns the expansion coefficients (eddies and zonal components combined),
  as well as saved filename if save is true
  '''

  save  = kargs.get('save',True)   # create file

  print('- Expansion coefficients -')
  print (' - loading parameters from Hough functions file:\n    %s'%hfile)
  if hfile.endswith('.npz'):
    ohfile=np.load(hfile)
    three,nN,nmm,nk,nLat=ohfile['HOUGHs_UVZ'].shape
  else: # netcdf
    # loading dimensions:
    nc=netCDF4.Dataset(hfile)
    nN=nc.dimensions['max_zonal_wave_number_and_zonal_mean'].size
    nk=nc.dimensions['number_equivalent_heights'].size
    nc.close()

  print (' - loading vertical structure functions:\n    %s'%vfile)
  if vfile.endswith('.npz'):
    ovfile=np.load(vfile)
    ws0=eval(ovfile['ws0'][()]) # ws0 is the string True/False

    hk    = ovfile['hk']
    Gn    = ovfile['Gn']
    p_new = ovfile['plev']
  else: # netcdf
    nc=netCDF4.Dataset(vfile)
    ws0=eval(nc.ws0)

    hk    = nc.variables['hk'][:]
    Gn    = nc.variables['Gn'][:]
    p_new = nc.variables['plev'][:]
    nc.close()

  if 'u' in data: # zonal wind, meridional wind, geopotential
    a_nk=prep(data['u'],p_new,nk,Gn,hk,nN,'zonal wind')
    b_nk=prep(data['v'],p_new,nk,Gn,hk,nN,'meridional wind')
    c_nk=prep(data['z'],p_new,nk,Gn,hk,nN,'geopotential')
    var_nk = np.zeros((3,)+a_nk.shape,dtype=cType)

    # Storing vertical and Fourier transforms of u, v and PHI:
    var_nk[0]=a_nk
    var_nk[1]=b_nk
    var_nk[2]=c_nk

    Lat=data['u']['lat']
    type='uvz'

  elif 'I1' in data:
    a_nk=prep(data['I1'],p_new,nk,Gn,hk,nN,'I1')
    b_nk=prep(data['I2'],p_new,nk,Gn,hk,nN,'I2')
    var_nk = np.zeros((3,)+a_nk.shape,dtype=cType)

    var_nk[0]=a_nk
    var_nk[1]=b_nk

    Lat=data['I1']['lat']
    type='I'

  elif 'J3' in data:
    c_nk=prep(data['J3'],p_new,nk,Gn,hk,nN,'J3')
    var_nk = np.zeros((3,)+c_nk.shape,dtype=cType)

    var_nk[2]=c_nk

    Lat=data['J3']['lat']
    type='J'


  # Hough transforms -------------------------------------------------
  print(' - loading Hough vector functions:\n    %s'%hfile)
  if hfile.endswith('.npz'):
    HOUGHs_UVZ = ohfile['HOUGHs_UVZ']
  else:
    nc=netCDF4.Dataset(hfile)
    HOUGHs_UVZ=nc.variables['HOUGHs_UVZ_real'][:]+1j*nc.variables['HOUGHs_UVZ_imag'][:]
    nc.close()

  var_nlk=transforms.hough(HOUGHs_UVZ,var_nk,Lat,ws0)

  if save:
    if type=='uvz':
      data=dict(w_nlk=var_nlk)
    elif type=='I':
      data=dict(i_nlk=var_nlk)
    elif type=='J':
      data=dict(j_nlk=var_nlk)

    fsave=save_out(data,**kargs)
    return var_nlk,fsave
  else:
    return var_nlk


def prep(data,p_new,nk,Gn,hk,nN,dataLabel):
  u   = data['v']
  Lat = data['lat']
  Lon = data['lon']
  P   = data['P']

  u_k=transforms.vertical(u,hk,nk,Gn,P,p_new,dataLabel)

  print (' - %s - Fourier transform'%dataLabel)
  # Fourier transform
  U_nk = np.fft.fft(u_k,n=None, axis=2)   # U_nk is the Fourier Transform of u_k along dimension 2 (i.e along longitude)
  U_nk = U_nk / Lon.size       # Scale the fft so that it is not a function of the length of input vector

  # Retaining the first nN (see hough_functions) zonal wave numbers
  # U_nk has dimensions U_nk(m,Lat,nN=Lon,t)
  U_nk = U_nk[:,:,:nN,:]   # First nN wavenumbers with zonal mean

  return U_nk


def save_out(data,**kargs):
  label=kargs.get('label','out')
  format=kargs.get('format','nc') # nc or npz
  attrs=kargs.get('attrs',{})

  import platform
  import sys
  attrs['platform']      = platform.platform()
  attrs['environment']   = 'python'
  attrs['version']       = sys.version.replace('\n','')
  attrs['version_scipy'] = scipy.__version__
  attrs['version_numpy'] = np.__version__

  if   'w_nlk' in data: label2='w'
  elif 'i_nlk' in data: label2='i'
  elif 'j_nlk' in data: label2='j'

  fsave='%s_%s_nlk.%s'%(label,label2,format)

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

  if 'w_nlk' in data:
    vname='w_nlk'
    vlname='Expansion coefficients of dependent variable vector u, v, z'
  elif 'i_nlk' in data:
    vname='i_nlk'
    vlname='Expansion coefficients of nonlinear term vector due to wind field'
  elif 'j_nlk' in data:
    vname='j_nlk'
    vlname='Expansion coefficients of nonlinear term due to mass field'

  v=data[vname]

  nc=netCDF4.Dataset(fname,'w',file_format='NETCDF4_CLASSIC')

  # dimensions:
  nk,nN,nL,nTime=v.shape
  nc.createDimension('number_equivalent_heights',nk)
  nc.createDimension('max_zonal_wave_number',nN)
  nc.createDimension('total_meridional_modes',nL) # LG+LR
  nc.createDimension('time',nTime)

  # variables:
  dim='number_equivalent_heights','max_zonal_wave_number','total_meridional_modes','time'
  vr=nc.createVariable(vname+'_real',dType,dim)
  vr.long_name=vlname+' (real)'
  vi=nc.createVariable(vname+'_imag',dType,dim)
  vi.long_name=vlname+' (imag)'

  # global attributes:
  import datetime
  nc.date=datetime.datetime.now().isoformat(' ')
  for k in attrs.keys():
    setattr(nc,k,attrs[k])

  # fill variables
  nc.variables[vname+'_real'][:]=v.real
  nc.variables[vname+'_imag'][:]=v.imag
  nc.close()
