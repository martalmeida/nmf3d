'''
Hough vector functions as described in Swarztrauber and Kasahara (1985)

References:
  A. Kasahara (1984). The Linear Response of a Stratified Global Atmosphere to
  Tropical Thermal Forcing, J. Atmos. Sci., 41(14). 2217--2237.
  doi: 10.1175/1520-0469(1984)041<2217:TLROAS>2.0.CO;2

  P. N. Swarztrauber and A. Kasahara (1985). The vector harmonic analysis of
  Laplace's tidal equations, SIAM J. Sci. Stat. Comput, 6(2), 464-491.
  doi: 10.1137/0906033

  A. Kasahara (1976). Normal modes of ultralong waves in the atmosphere, Mon.
  Weather Rev., 104(6), 669-690. doi: 10.1175/1520-0493(1976)1042.0.CO;2

  Y. Shigehisa (1983). Normal Modes of the Shallow Water Equations for Zonal
  Wavenumber Zero, J. Meteorol. Soc. Jpn., 61(4), 479-493.
  doi: 10.2151/jmsj1965.61.4_479

  A. Kasahara (1978). Further Studies on a Spectral Model of the Global
  Barotropic Primitive Equations with Hough Harmonic Expansions, J. Atmos.
  Sci., 35(11), 2043-2051. doi: 10.1175/1520-0469(1978)0352.0.CO;2
'''

from . import constants as const
from . import hvf_baroclinic
import numpy as np

dType='d'
#cType=np.complex128

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

  Returns baroclinic (data_b) and barotropic (data_B) data dicts and well as saved
  filenames (fsave_b and fsvae_B) if save is True. If hk[0] is not inf (ws0 is False,
  see vertical_structuree) no barotropic component is returned, ie:
  -- if hk[0] is inf
     - return data_b,data_B,fsave_b,fsave_B (if save is True)
     - return data_b,data_B (if save is False)
  -- if hk[0] is not inf
     - returns data_b,fsave (if save is True)
     - return data_b (if save is False)
  '''

  save  = kargs.get('save',True)   # create file

  if latType=='linear': default_dlat=1.5
  elif latType=='gaussian': default_dlat=128
  dlat=kargs.get('dlat',default_dlat)

  # keep nk as global attribute of nc files
  nk=hk.size
  attrs=kargs.get('attrs',{})
  attrs['nk']= np.int8(nk)
  kargs['attrs']=attrs

  params=dict(M=M,nLR=nLR,nLG=nLG,NEH='unk',dlat=dlat,latType=latType) # just for saving
  if np.isinf(hk[0]): # ws0 True
    # baroclinic:
    data_b,trunc,x=hvf_baroclinic.hvf(hk[1:],M,nLR,nLG,latType,dlat)
    params['NEH']=hk.size-1
    if save: fsave_b=save_out(data_b,'baroclinic',params,**kargs)

    # barotropic:
    from . import hvf_barotropic
    data_B=hvf_barotropic.hvf_bar(nLR,nLG,M,trunc,x)
    params['NEH']=1
    if save: fsave_B=save_out(data_B,'barotropic',params,**kargs)

    if save:  return data_b,data_B,fsave_b,fsave_B
    else: return data_b,data_B
  else:
    data_b,trunc,x=hvf_baroclinic.hvf(hk,M,nLR,nLG,latType,dlat)
    params['NEH']=hk.size
    if save:
      fsave=save_out(data_b,'ws0False',params,**kargs)
      return data_b,fsave
    else: return data_b


def save_out(data,tag,params,**kargs):
  label=kargs.get('label','out')
  format=kargs.get('format','nc') # nc or npz
  attrs=kargs.get('attrs',{})

  import platform
  import sys
  import scipy
  attrs['platform']=platform.platform()
  attrs['environment']='python'
  attrs['version']=sys.version
  attrs['version_scipy']=scipy.__version__
  attrs['version_numpy']=np.__version__

  fsave='%s_hvf_M%d_nLR%d_nLG%d_NEH%d_dlat%s%s_%s.%s'%(label,params['M'],params['nLR'],params['nLG']*2,
                                                       params['NEH'],params['dlat'],params['latType'],tag,format)

  print('saving %s'%fsave)
  if format=='npz':
    data.update(attrs)
    np.savez(fsave, **data)
  elif format=='nc':
    if tag=='barotropic':
      save_nc_bar(fsave,data,**attrs)
    else:
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
  M,L,NEH,lat=data['HOUGH_UVZ'].shape[1:]
  quarter_nLG=data['WEST_G_sy'].shape[1]
  half_nLR=data['WEST_R_sy'].shape[1]
  Np1=data['WEST_R_0_sy'].shape[0]

  nc.createDimension('components_uvz',3)              # components uvz
  nc.createDimension('max_zonal_wave_number',M)       # max zonal wave number (M)
  nc.createDimension('number_meridional_modes',L)     # total number meridional modes (L, must be even!)
  nc.createDimension('lat',lat)                       # n lats
  nc.createDimension('number_equivalent_heights',NEH) # number equivalent heights (NEH)

  nc.createDimension('quarter_number_gravitical_modes',quarter_nLG)
  nc.createDimension('half_number_rossby_modes',half_nLR)
  nc.createDimension('Np1',Np1)

  # Note that L=NG+NR=4*quarter_number_gravitical_modes+2*half_number_rossby_modes


  # variables:
  #   hough:
  k='HOUGH_UVZ'
  dim='components_uvz','max_zonal_wave_number','number_meridional_modes','number_equivalent_heights','lat'
  v=nc.createVariable(k+'_real',dType,dim)
  v.long_name='hough functions - eddies (real)'
  v=nc.createVariable(k+'_imag',dType,dim)
  v.long_name='hough functions - eddies (imag)'

  k='HOUGH_0_UVZ'
  dim='components_uvz','number_meridional_modes','number_equivalent_heights','lat'
  v=nc.createVariable(k+'_real',dType,dim)
  v.long_name='hough functions - zonal mean (real)'
  v=nc.createVariable(k+'_imag',dType,dim)
  v.long_name='hough functions - zonal mean (imag)'


  #   westward - eddies:
  dim='max_zonal_wave_number','quarter_number_gravitical_modes','number_equivalent_heights'

  k='WEST_G_sy'
  v=nc.createVariable(k,data[k].dtype,dim)
  v.long_name='frequencies of the symmetric westward gravity waves - eddies'

  k='WEST_G_asy'
  v=nc.createVariable(k,data[k].dtype,dim)
  v.long_name='frequencies of the antisymmetric westward gravity waves - eddies'

  dim='max_zonal_wave_number','half_number_rossby_modes','number_equivalent_heights'

  k='WEST_R_sy'
  v=nc.createVariable(k,data[k].dtype,dim)
  v.long_name='frequencies of the symmetric westward Rossby waves - eddies'

  k='WEST_R_asy'
  v=nc.createVariable(k,data[k].dtype,dim)
  v.long_name='frequencies of the antisymmetric westward Rossby waves - eddies'

  #   eastward - eddies:
  dim='max_zonal_wave_number','quarter_number_gravitical_modes','number_equivalent_heights'

  k='EAST_G_sy'
  v=nc.createVariable(k,data[k].dtype,dim)
  v.long_name='frequencies of the symmetric eastward gravity waves - eddies'

  k='EAST_G_asy'
  v=nc.createVariable(k,data[k].dtype,dim)
  v.long_name='frequencies of the antisymmetric eastward gravity waves - eddies'


  #   westward - zonal mean:
  dim='quarter_number_gravitical_modes','number_equivalent_heights'

  k='WEST_G_0_sy'
  v=nc.createVariable(k,data[k].dtype,dim)
  v.long_name='frequencies of the symmetric westward gravity waves - zonal mean'

  k='WEST_G_0_asy'
  v=nc.createVariable(k,data[k].dtype,dim)
  v.long_name='frequencies of the antisymmetric westward gravity waves - zonal mean'


  dim='half_number_rossby_modes','number_equivalent_heights'

  k='WEST_R_0_sy'
  v=nc.createVariable(k,data[k].dtype,dim)
  v.long_name='frequencies of the symmetric westward Rossby waves - zonal mean'

  k='WEST_R_0_asy'
  v=nc.createVariable(k,data[k].dtype,dim)
  v.long_name='frequencies of the antisymmetric westward Rossby waves - zonal mean'

  #   eastward - zonal mean:
  dim='quarter_number_gravitical_modes','number_equivalent_heights'

  k='EAST_G_0_sy'
  v=nc.createVariable(k,data[k].dtype,dim)
  v.long_name='frequencies of the symmetric eastward gravity waves - zonal mean'

  k='EAST_G_0_asy'
  v=nc.createVariable(k,data[k].dtype,dim)
  v.long_name='frequencies of the antisymmetric eastward gravity waves - zonal mean'


  # global attributes:
  import datetime
  nc.date=datetime.datetime.now().isoformat(' ')
  for k in attrs.keys():
    setattr(nc,k,attrs[k])


  # fill variables
  for k in data.keys():
    if k.startswith('HOUGH'):
      if debug: print('filling %s  shape=%s %s'%(k+'_real', str(nc.variables[k+'_real'].shape),str(data[k].shape)))
      nc.variables[k+'_real'][:]=data[k].real

      if debug: print('filling %s  shape=%s %s'%(k+'_imag', str(nc.variables[k+'_imag'].shape),str(data[k].shape)))
      nc.variables[k+'_imag'][:]=data[k].imag

    else:
      if debug: print('filling %s  shape=%s %s'%(k, str(nc.variables[k].shape),str(data[k].shape)))
      nc.variables[k][:]=data[k]

  nc.close()


def save_nc_bar(fname,data,**attrs):
  debug=0

  import netCDF4

  import os
  if os.path.isfile(fname):
    os.unlink(fname)

  nc=netCDF4.Dataset(fname,'w',file_format='NETCDF4_CLASSIC')

  # dimensions:
  M,L,lat=data['HOUGH_UVZ'].shape[1:]
  nLR=data['SIGMAS'].shape[1]

  nc.createDimension('components_uvz',3)              # components uvz
  nc.createDimension('max_zonal_wave_number',M)       # max zonal wave number (M)
  nc.createDimension('number_meridional_modes',L)     # total number meridional modes (L, must be even!)
  nc.createDimension('lat',lat)                       # n lats
  nc.createDimension('number_Rossby_modes',nLR)       # total number of (west) Rossby modes (must be even!)


  # variables:
  #   hough and hough to reconstr.:
  k='HOUGH_UVZ'
  dim='components_uvz','max_zonal_wave_number','number_meridional_modes','lat'
  v=nc.createVariable(k+'_real',dType,dim)
  v.long_name='hough functions - eddies (real)'
  v=nc.createVariable(k+'_imag',dType,dim)
  v.long_name='hough functions - eddies (imag)'

  k='HOUGH_UVZ_2rec'
  v=nc.createVariable(k+'_real',dType,dim)
  v.long_name='hough functions for reconstruction - eddies (real)'
  v=nc.createVariable(k+'_imag',dType,dim)
  v.long_name='hough functions for reconstruction - eddies (imag)'

  k='HOUGH_0_UVZ'
  dim='components_uvz','number_meridional_modes','lat'
  v=nc.createVariable(k,dType,dim)
  v.long_name='hough functions - zonal mean'

  k='HOUGH_0_UVZ_2rec'
  v=nc.createVariable(k,dType,dim)
  v.long_name='hough functions for reconstruction - zonal mean'

  #   sigmas:
  k='SIGMAS'
  dim='max_zonal_wave_number','number_Rossby_modes'
  v=nc.createVariable(k,dType,dim)
  v.long_name='Haurwitz frequencies'


  # global attributes:
  import datetime
  nc.date=datetime.datetime.now().isoformat(' ')
  for k in attrs.keys():
    setattr(nc,k,attrs[k])


  # fill variables
  for k in data.keys():
    if k.startswith('HOUGH_UVZ'):
      if debug: print('filling %s  shape=%s %s'%(k+'_real', str(nc.variables[k+'_real'].shape),str(data[k].shape)))
      nc.variables[k+'_real'][:]=np.real(data[k])

      if debug: print('filling %s  shape=%s %s'%(k+'_imag', str(nc.variables[k+'_imag'].shape),str(data[k].shape)))
      nc.variables[k+'_imag'][:]=np.imag(data[k])

    else:
      if debug: print('filling %s  shape=%s %s'%(k, str(nc.variables[k].shape),str(data[k].shape)))
      nc.variables[k][:]=data[k]

  nc.close()
