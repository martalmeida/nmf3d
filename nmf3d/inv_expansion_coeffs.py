from . import constants as const
import numpy as np
import scipy.interpolate
import netCDF4

dType='d'
cType='complex128'

def calc(vfile,hfile,wfile,zi,mi,vi,pl,lon,**kargs):
  '''
  Inverse expansion coefficients
  Compute the inverse of Vertical, Fourier and Hough transforms.
  Recovers the zonal and meridional wind, and geopotential perturbation
  (from the reference geopotential) in the physical space, for the chosen
  set of modes. Uses equivalent heights, vertical structure functions
  Hough functions and complex expansion coefficients.

  vfile: file with the equivalent heights (hk) and vertical structure functions (Gn)
  hfile: file with the Hough functions (HOUGHs_UVZ)
  wfile: file with the expansion coefficients (w_nlk)
  zi:    wavenumber indices, zi=0 (Zonal mean) and zi>0 (Eddies)
  mi:    meridional indices
  vi:    vertical indices, vi=0 (Barotropic mode) and vi>0 (Baroclinic modes)
  pl:    pressure levels (hPa units)
  lon:   longitudes (deg units)
  kargs:
    uvz:   components of wind and mass fields to calculate
           - default [1 1 1] - calculate the zonal and meridional wind and the geopotential (u, v, z)
    save, create file [true]
    format, file format: [nc] or npz
    attrs, attributes to save [{}]
    label, filename to save, without extension ['outinv_uvz']

  Returns the reconstructed  zonal and meridional wind, and geopotential perturbation,
  as well as saved filename if save is true
  '''

  uvz  = kargs.get('uvz',[1,1,1])
  save = kargs.get('save',True)

  # The wavenumber, meridional and vertical indices
  ZI = np.asarray(zi)
  MI = np.asarray(mi)-1
  VI = np.asarray(vi)

  pl = np.asarray(pl)
  lon = np.asarray(lon)

  lon=lon*np.pi/180 # lon in radians

  print(' - loading vertical structure functions:\n    %s'%vfile)
  if vfile.endswith('.npz'):
    ovfile=np.load(vfile)
    ws0=eval(ovfile['ws0'][()]) # ws0 is the string True/False

    hk    = ovfile['hk']
    Gn    = ovfile['Gn']
    p_new = ovfile['plev']
  else:
    nc=netCDF4.Dataset(vfile)
    ws0=eval(nc.ws0)

    hk    = nc.variables['hk'][:]
    Gn    = nc.variables['Gn'][:]
    p_new = nc.variables['plev'][:]
    nc.close()


  print(' - loading Hough vector functions:\n    %s'%hfile);
  if hfile.endswith('.npz'):
    ohfile=np.load(hfile)
    HOUGHs_UVZ = ohfile['HOUGHs_UVZ']
    lat = ohfile['lat']
  else:
    nc=netCDF4.Dataset(hfile)
    HOUGHs_UVZ=nc.variables['HOUGHs_UVZ_real'][:]+1j*nc.variables['HOUGHs_UVZ_imag'][:]
    lat=nc.variables['lat'][:]
    nc.close()

  print(' - loading expansion coefficients:\n    %s'%wfile);
  if wfile.endswith('.npz'):
    owfile=np.load(wfile)
    w_nlk = owfile['w_nlk']
  else:
    nc=netCDF4.Dataset(wfile)
    w_nlk=nc.variables['w_nlk_real'][:]+1j*nc.variables['w_nlk_imag'][:]
    nc.close()

  three,nN,nmm,nk,nLat=HOUGHs_UVZ.shape
  hk=hk[:nk]
  nT=w_nlk.shape[-1]

  if ws0: hk[0]=1

  # Interpolates vertical structure functions (Gn) for level p with cubic spline
  plev=p_new
  plev[0]=const.ps # surface max pressure, to avoid extrap!
  Gn_interpS=np.zeros((nk,pl.size),Gn.dtype)
  for i in range(nk):
    Gn_interpS[i] = scipy.interpolate.UnivariateSpline(plev[::-1],Gn[i][::-1],s=0)(pl*100) # plev pl*100 in Pa

  # Permuting dimensions of Fourier-Hough transforms for convenience
  w_nlk = np.transpose(w_nlk,(1,2,0,3)) # % shape w_nlk  = n,l,k,t

  if max(ZI)>nN:
    raise Exception('maximum wavenumber was exceeded')

  if max(VI)>nk:
    raise Exception('maximum vertical index was exceeded')

  # variables for the horizontal wind and mass fields expansion into 3-D normal modes
  un = np.zeros((nN, nLat, lon.size),dtype=dType) # size u  = n,lat,lon
  vn = np.zeros((nN, nLat, lon.size),dtype=dType)
  zn = np.zeros((nN, nLat, lon.size),dtype=dType)

  u = np.zeros((nT,pl.size,nLat,lon.size),dtype=dType) # size u  = nT,npL,lat,lon
  v = np.zeros((nT,pl.size,nLat,lon.size),dtype=dType)
  z = np.zeros((nT,pl.size,nLat,lon.size),dtype=dType)

  select_nlk = np.zeros((nN,nmm,nk))
  ZI,MI,VI=np.meshgrid(ZI,MI,VI)
  select_nlk[ZI,MI,VI]=1

  print(' - computing')
  lon.shape=1,lon.size
  for ip in range(pl.size): # loop over p levels
    Gn = Gn_interpS[:,ip]
    for it in range(nT): # loop over times
      A_w_nlk = select_nlk*w_nlk[:,:,:,it]

      # Replicating variables into arrays with matching sizes for
      # elementwise array multiplications (element-by-element). The sizes
      # must mach for dimensions (n,l,k)
      A_h_k = np.transpose(np.tile(hk[:,np.newaxis,np.newaxis],[1,nN,nmm]),[1,2,0]) # size A_h_k  = n,l,k
      A_Gn  = np.transpose(np.tile(Gn[:,np.newaxis,np.newaxis],[1,nN,nmm]),[1,2,0]) # size A_Gn   = n,l,k

      nW = np.arange(nN)
      nW.shape=nW.size,1

      tmp=np.exp(const.I*np.matmul(nW,lon))
      A_Fourier=np.tile(tmp[:,:,np.newaxis,np.newaxis],[1,1,nmm,nk]) # size A_Fourier = n,lon,l,k

      for la in range(nLat):
        for lo in range(lon.size):
          # summing over chosen vertical and meridional modes
          un[:,la,lo]=(A_w_nlk*np.sqrt(const.g*A_h_k)*A_Gn*HOUGHs_UVZ[0,:,:,:,la]*A_Fourier[:,lo,:,:]).sum(-1).sum(-1).real*2
          vn[:,la,lo]=(A_w_nlk*np.sqrt(const.g*A_h_k)*A_Gn*HOUGHs_UVZ[1,:,:,:,la]*A_Fourier[:,lo,:,:]).sum(-1).sum(-1).real*2
          zn[:,la,lo]=(A_w_nlk*        const.g*A_h_k *A_Gn*HOUGHs_UVZ[2,:,:,:,la]*A_Fourier[:,lo,:,:]).sum(-1).sum(-1).real*2

      u[it,ip]=un[1:].sum(0)+0.5*un[0]
      v[it,ip]=vn[1:].sum(0)+0.5*vn[0]
      z[it,ip]=zn[1:].sum(0)+0.5*zn[0]

  lon.shape=lon.size
  data=dict(lon=lon*180/np.pi,lat=lat)
  if uvz[0]: data['u']=u
  if uvz[1]: data['v']=v
  if uvz[2]: data['z']=z

  if save:
    # add some input data as attributes:
    attrs=kargs.get('attrs',{})
    attrs.update(dict(zi=zi,mi=mi,vi=vi,pl=pl,uvz=uvz))
    kargs['attrs']=attrs

    fsave=save_out(data,**kargs)
    return data,fsave
  else:
    return data


def save_out(data,**kargs):
  label=kargs.get('label','outinv_uvz')
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

  fsave='%s.%s'%(label,format)

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

  addu='u' in data
  addv='v' in data
  addz='z' in data

  nc=netCDF4.Dataset(fname,'w',file_format='NETCDF4_CLASSIC')

  # dimensions:
  if   addu: tmp=data['u']
  elif addv: tmp=data['v']
  elif addz: tmp=data['z']

  if addu or addv or addz:
    nT,nLev,nLat,nLon=tmp.shape

    nc.createDimension('time',nT)
    nc.createDimension('pressure_levels',nLev)
    nc.createDimension('lat',nLat)
    nc.createDimension('lon',nLon)

    dim='time','pressure_levels','lat','lon'
  else:
    nc.createDimension('lat',data['lat'].size)
    nc.createDimension('lon',data['lon'].size)

  # variables:
  if addu:
    v=nc.createVariable('u',dType,dim)
    v.long_name='reconstructed zonal wind'

  if addv:
    v=nc.createVariable('v',dType,dim)
    v.long_name='reconstructed meridional wind'

  if addz:
    v=nc.createVariable('z',dType,dim)
    v.long_name='reconstructed geopotential perturbation'

  v=nc.createVariable('lon',dType,('lon',))
  v.long_name='longitude'
  v=nc.createVariable('lat',dType,('lat',))
  v.long_name='latitude'

  # global attributes:
  import datetime
  nc.date=datetime.datetime.now().isoformat(' ')
  for k in attrs.keys():
    setattr(nc,k,attrs[k])

  # fill variables:
  if addu: nc.variables['u'][:]=data['u']
  if addv: nc.variables['v'][:]=data['v']
  if addz: nc.variables['z'][:]=data['z']
  nc.variables['lon'][:]=data['lon']
  nc.variables['lat'][:]=data['lat']
  nc.close()
