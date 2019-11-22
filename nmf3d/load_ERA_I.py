import numpy as np
import netCDF4
from . import constants as const

dType='d'

def load_once(f,fref=None,height=False):
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
    if fref.endswith('.npz'):
      q=np.load(fref)['phi0'].astype(dType)
    else:
      q=np.loadtxt(fref)[0].astype(dType)

    print('    - subtracting reference')
    for i in range(P.size): v[:,i,:,:]-=q[i]

  if height:
    print('    - geopotential height --> geopotential')
    v/=const.g

  return dict(lon=lon,lat=lat,P=P,v=v)


def load(fu,fv,fz,fzref,height):
  res={}

  def shortname(f):
    import os
    return '....'+os.path.join(os.path.basename(os.path.dirname(f)),os.path.basename(f))

  print('loading u : %s'%shortname(fu))
  res['u']=load_once(fu)
  print('loading v : %s'%shortname(fv))
  res['v']=load_once(fv)
  print('loading z : %s'%shortname(fz))
  res['z']=load_once(fz,fzref,height)
  return res
