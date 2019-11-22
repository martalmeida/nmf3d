'''
Hough vector functions as described in Swarztrauber and Kasahara (1985)

References:
  A. Kasahara (1976). Normal modes of ultralong waves in the atmosphere, Mon.
  Weather Rev., 104(6), 669-690. doi: 10.1175/1520-0493(1976)1042.0.CO;2

  A. Kasahara (1978). Further Studies on a Spectral Model of the Global
  Barotropic Primitive Equations with Hough Harmonic Expansions, J. Atmos.
  Sci., 35(11), 2043-2051. doi: 10.1175/1520-0469(1978)0352.0.CO;2

  Y. Shigehisa (1983). Normal Modes of the Shallow Water Equations for Zonal
  Wavenumber Zero, J. Meteorol. Soc. Jpn., 61(4), 479-493.
  doi: 10.2151/jmsj1965.61.4_479

  A. Kasahara (1984). The Linear Response of a Stratified Global Atmosphere to
  Tropical Thermal Forcing, J. Atmos. Sci., 41(14). 2217--2237.
  doi: 10.1175/1520-0469(1984)041<2217:TLROAS>2.0.CO;2

  P. N. Swarztrauber and A. Kasahara (1985). The vector harmonic analysis of
  Laplace's tidal equations, SIAM J. Sci. Stat. Comput, 6(2), 464-491.
  doi: 10.1137/0906033
'''

from . import constants as const
from . import hvf_baroclinic
import numpy as np

dType='d'
cType=np.complex128

def hvf(hk,M=42,nLR=40,nLG=20,latType='linear',**kargs):
  '''
  Hough vector functions
  The total number of the Gravity modes will be 2*nLG=nLG(east gravity)+nLG(west gravity)
  Part I: The frequencies and the Hough functions are computed for zonal wave number m = 0
  Part II: The frequencies and the Hough functions are computed for zonal wave numbers m > 0

  M, maximum zonal wave number used in the expansion: m=0,1,...,M
  nLR, total number of (west) Rossby modes used in the expansion (should be even)
  nLG , half the number of Gravity modes used in the expansion (should be even)
  latType, latitude type: linear (default, equally spaced) or gaussian
  kargs:
    dlat, latitude spacing if latType is linear (default is 1.5, ie, 121 points) or
          number of gaussian lats if latType is gaussian (default is 128, corresponding
          to a spectral truncature of T85)
    save, create file [True]
    format, file format: [nc] or npz
    attrs, attributes to save [{}]
    label, start of the saved filename ['out']

  Returns concatenated baroclinic and barotropic data as well as saved filename
  if save is True. If hk[0] is not inf (ws0 is False, see vertical_structure)
  no barotropic component is returned
  '''

  save  = kargs.get('save',True)   # create file

  if latType=='linear': default_dlat=1.5
  elif latType=='gaussian': default_dlat=128
  dlat=kargs.get('dlat',default_dlat)

  params=dict(M=M,nLR=nLR,nLG=nLG,NEH=hk.size,dlat=dlat,latType=latType) # just for saving

  L=nLR + 2*nLG

  if np.isinf(hk[0]): # ws0 True
    # baroclinic:
    data_b,trunc,x=hvf_baroclinic.hvf(hk[1:],M,nLR,nLG,latType,dlat)

    # barotropic:
    from . import hvf_barotropic
    data_B=hvf_barotropic.hvf_bar(nLR,nLG,M,trunc,x)

    # concatenate barotropic and baroclinic, Hough functions:
    # zonal:
    HOUGHs_0_UVZ=np.zeros((3,L,hk.size,x.size),dtype=cType)
    HOUGHs_0_UVZ[:,:,0]  = data_B['HOUGH_0_UVZ']
    HOUGHs_0_UVZ[:,:,1:] = data_b['HOUGH_0_UVZ']

    # eddies:
    HOUGHs_m_UVZ=np.zeros((3,M,L,hk.size,x.size),dtype=cType)
    HOUGHs_m_UVZ[:,:,:,0] = data_B['HOUGH_UVZ']
    HOUGHs_m_UVZ[:,:,:,1:] = data_b['HOUGH_UVZ']

    # concatenate zonal and eddies:
    HOUGHs_UVZ=np.zeros((3,M+1,L,hk.size,x.size),dtype=cType)
    HOUGHs_UVZ[:,0]  = HOUGHs_0_UVZ
    HOUGHs_UVZ[:,1:] = HOUGHs_m_UVZ

    # concatenate barotropic and baroclinic, frequencies:
    # zonal:
    FREQs_0=np.zeros((L,hk.size),dtype=dType)
    FREQs_0[:,1:]=data_b['FREQS_0']

    # eddies:
    sigmas=np.zeros((M,L))*np.nan
    sigmas[:,-nLR:]=data_B['SIGMAS']
    FREQs_m = np.zeros((M,L,hk.size),dtype=dType)
    FREQs_m[:,:,0]  = sigmas
    FREQs_m[:,:,1:] = data_b['FREQS_m']

    # concatenate zonal and eddies:
    FREQs= np.zeros((M+1,L,hk.size),dtype=dType)
    FREQs[0]  = FREQs_0
    FREQs[1:] = FREQs_m

    # data to store:
    data=dict(HOUGHs_UVZ=HOUGHs_UVZ,FREQs=FREQs)

    if save:
      fsave=save_out(data,'ws0True',params,**kargs);
      return data,fsave
    else: return data

  else:
    data_b,trunc,x=hvf_baroclinic.hvf(hk,M,nLR,nLG,latType,dlat)

    # concatenate zonal and eddies, Hough functions and frequencies:
    HOUGHs_UVZ=np.zeros((3,M+1,L,hk.size,x.size),dtype=cType)
    HOUGHs_UVZ[:,0]  = data_b['HOUGH_0_UVZ']
    HOUGHs_UVZ[:,1:] = data_b['HOUGH_UVZ']

    FREQs=np.zeros((M+1,L,hk.size),dtype=dType)
    FREQs[0]  = data_b['FREQS_0']
    FREQs[1:] = data_b['FREQS_m']

    # data to store:
    data=dict(HOUGHs_UVZ=HOUGHs_UVZ,FREQs=FREQs)

    if save:
      fsave=save_out(data,'ws0False',params,**kargs)
      return data,fsave
    else: return data


def save_out(data,tag,params,**kargs):
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

  fsave='%s_hvf_M%d_nLR%d_nLG%d_NEH%d_dlat%s%s_%s.%s'%(label,params['M'],params['nLR'],params['nLG']*2,
                                                       params['NEH'],params['dlat'],params['latType'],tag,format)

  print('saving %s'%fsave)
  if format=='npz':
    data.update(attrs)
    np.savez(fsave, **data)
  elif format=='nc':
    save_nc(fsave,data,**attrs)
  else: print('Unknown format, use nc or npz')

  return fsave


def save_nc(fname,data,**attrs):
  debug=0
  import netCDF4

  import os
  if os.path.isfile(fname):
    os.unlink(fname)

  nc=netCDF4.Dataset(fname,'w',file_format='NETCDF4_CLASSIC')

  # dimensions:
  M_,L,NEH,lat=data['HOUGHs_UVZ'].shape[1:]

  nc.createDimension('components_uvz',3)              # components uvz
  nc.createDimension('max_zonal_wave_number_and_zonal_mean',M_) # max zonal wave number (M)
  nc.createDimension('number_meridional_modes',L)     # total number meridional modes (L, must be even!)
  nc.createDimension('lat',lat)                       # n lats
  nc.createDimension('number_equivalent_heights',NEH) # number equivalent heights (NEH)

  # variables:
  #   hough:
  k='HOUGHs_UVZ'
  dim='components_uvz','max_zonal_wave_number_and_zonal_mean','number_meridional_modes','number_equivalent_heights','lat'
  v=nc.createVariable(k+'_real',dType,dim)
  v.long_name='hough functions - real'
  v=nc.createVariable(k+'_imag',dType,dim)
  v.long_name='hough functions - imag'

  # frequencies:
  k='FREQs'
  dim='max_zonal_wave_number_and_zonal_mean','number_meridional_modes','number_equivalent_heights'
  v=nc.createVariable(k,dType,dim)
  v.long_name='frequencies'

  # global attributes:
  import datetime
  nc.date=datetime.datetime.now().isoformat(' ')
  for k in attrs.keys():
    setattr(nc,k,attrs[k])

  # fill variables
  nc.variables['HOUGHs_UVZ_real'][:]=data['HOUGHs_UVZ'].real
  nc.variables['HOUGHs_UVZ_imag'][:]=data['HOUGHs_UVZ'].imag
  nc.variables['FREQs'][:]=data['FREQs']
  nc.close()
