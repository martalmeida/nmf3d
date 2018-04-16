from . import constants as const
from scipy.interpolate import splrep,splev
import scipy.misc
import scipy.linalg
import scipy.special
import numpy as np

dType='d'

def stability(Tref,Gp):
  '''
  Static stability in the sigma system (eq. (A3) in Kasahara (1984))
  Derivative, by finite differences, of reference temperature (Tref) with respect to logarithm of p/ps (dT0_dLn_pps).
  '''

  pps = (Gp+1)/2; # p/Ps
  dTref_dLn_pps = np.zeros(Gp.size,dtype=Tref.dtype);

  # delta s:
  Ds = np.diff(np.log(pps))

  # forward differences (1st order):
  dTref_dLn_pps[0] = (Tref[1]- Tref[0]) / Ds[0];

  # Centred differences (2st order)
  for k in range(1,Gp.size-1):
      dTref_dLn_pps[k] = (1/(Ds[k-1]*Ds[k]*(Ds[k-1]+Ds[k]))) * (Ds[k-1]**2*Tref[k+1]-Ds[k]**2*Tref[k-1]-(Ds[k-1]**2-Ds[k]**2)*Tref[k])

  # Backward differences (1st order)
  dTref_dLn_pps[-1] = (Tref[-1]-Tref[-2]) / Ds[-1]

  # The static stability in the sigma system (Gamma0)
  return  (const.Qsi*Tref)/(1+Gp)-0.5/pps*dTref_dLn_pps


def calc_M(J,Gp,Gw,Gamma0,Tref1,ws0):
  '''
  Matrix Mij (Eq. (A12) in Kasahara (1984))

  '''

  GL=2*J-1

  # Normalized Associated Legendre functions
  m = 0 # order
  uP_s  = np.zeros((GL,J),dtype=Gp.dtype) # Unnormalized
  P_s   = np.zeros((GL,J),dtype=Gp.dtype) # Normalized
  P_s1  = np.zeros(J,dtype=Gp.dtype)      # Normalized (at sigma=1)

  # Looping over the degrees
  for n in range(0,J):
    n_m = scipy.misc.factorial(n-m)
    nm  = scipy.misc.factorial(n+m)
    # Looping over the gaussian points or levels (Gp)
    for z in  range(0,GL):
        [P0n,dP0n_dz] = scipy.special.lpmn(m, n, Gp[z])
        uP_s[z,n]     = P0n[m,n]
        P_s[z,n]      = (-1)**m * np.sqrt((n+0.5)*n_m/(nm)) * uP_s[z,n]

  # Looping over the degrees
  for n in range(0,J):
    # Legendre polynomials at sigma=1
    [P0n1,dP0n_dz1] = scipy.special.lpmn(m, n, 1.0)
    n_m = scipy.misc.factorial(n-m)
    nm  = scipy.misc.factorial(n+m)
    P_s1[n]         = (-1)**m * np.sqrt((n+0.5)*n_m/(nm)) * P0n1[0,n]


  # Derivative of Legendre polynomials with respect to sigma (d_P_ds)
  d_P_ds = np.zeros((GL,J),dtype=Gp.dtype);
  # The derivative of P_s(j=zero)=0 (all sigmas). Therefore, the index j of d_P_ds starts at j=1,
  # where the derivative of P_s(one) (all sigmas) is stored, and so on.
  for j in range(1,J):
    d_P_ds[:,j] = j*Gp/(Gp**2-1) * P_s[:,j] - j/(Gp**2 - 1) * np.sqrt((2.*(j)+1)/(2.*(j)-1)) * P_s[:,j-1];


  # Matrix Mij (Eq. (A12) in Kasahara (1984))
  M  = np.zeros((J,J),dtype=Gp.dtype);   # Initializing matrix Mij
  for i in range(0,J):
    for j in range(0,J):
      if ws0: # w=0 at surface
        M[i,j] = const.T00 * (((Gp+1.0)/Gamma0*d_P_ds[:,i]*d_P_ds[:,j])*Gw).sum()
      else:
        M[i,j] = const.T00 * (((Gp+1.0)/Gamma0*d_P_ds[:,i]*d_P_ds[:,j])*Gw).sum() + const.T00 * (2.0/Tref1*P_s1[i]*P_s1[j]);

  return M,P_s



def vse(Tprof,Plev,**kargs):
  '''Solves the vertical structure equation, returning the vertical
  structure functions and equivalent heights

  Tprof, reference temperature (mean vertical profile, K)
  Plev, corresponding pressure levels (Pa)
  kargs:
    ws0, If true (default false) the pressure vertical velocity  is zero at surface.
    n_leg, number of Legendre polynomials, Plev.size+20 by default
    nk, n functions to keep (cannot exceed the number of pressure levels, the default)
    save, create file [True]
    format, file format: [nc] or npz
    attrs, attributes to save [{}]
    label, start of the saved filename ['out']

  Example:
    import numpy as np
    import pylab as pl
    import netCDF4
    import vertical_structure

    a=np.load('tprof.npz');

    # netcdf output:
    Gn,hk,fname=vertical_structure.vse(a['Tprof'],a['Plev'],save=1,ws0=1)

    nc=netCDF4.Dataset(fname)
    plev=nc.variavles['plev'][:]
    semilogy(Gn(5,:),plev,'r.')

    # npz output:
    Gn,hk,fname=vertical_structure(a['Tprof'],a['Plev'],save=1,ws0=1,format='npz')

    b=np.load(fname);
    plot(b['hk'])

  '''

  ws0   = kargs.pop('ws0',False)
  n_leg = kargs.get('n_leg','auto')
  nk    = kargs.get('nk',Plev.size) # n functions to keep (max possible is Plev.size)
  save  = kargs.get('save',True)   # create file

  if n_leg=='auto': n_leg=Plev.size+20

  Tprof=Tprof.astype(dType)
  Plev=Plev.astype(dType)

  J=n_leg
  GL=2*J-1 # number of Gaussian levels

  # Gaussian levels (i.e. points (Gp)) and Gaussian weights (Gw)
  Gp,Gw = scipy.special.orthogonal.p_roots(GL)
  Gp=Gp[::-1] # flip Gp

  # Cubic spline interpolation of reference temperature from pressure to sigma levels
  Plev_s  = (Gp+1)*const.ps/2 # pressure levels that correspond to the chosen Gaussian sigma levels
  aux     = splrep(Plev, Tprof)
  Tprof_s = splev(Plev_s, aux, der=0, ext=0)

  # Reference Temperature at sigma=1, linearly extrapolated:
  Gp1 = 1
  Tprof_s1 = Tprof_s[1] + (Gp1-Gp[1])/(Gp[0]-Gp[1]) * (Tprof_s[0]-Tprof_s[1])

  # Static stability in the sigma system:
  Gamma0=stability(Tprof_s,Gp)#,GL)

  # Matrix Mij (Eq. (A12) in Kasahara (1984))
  M,P_s=calc_M(n_leg,Gp,Gw,Gamma0,Tprof_s1,ws0)

  # Eigenvectors and eigenvalues of matrix Mij
  useSvd=False
  if useSvd:
    #V,S,E = scipy.linalg.svd(M,lapack_driver='gesvd')
    V,S,E = scipy.linalg.svd(M)   # With svd the eigenvalues/eigenvectors are ordered
    S=S[::-1]
  else:
    S,V=scipy.linalg.eig(M,left=True,right=False)
    S=np.real(S)
    i=np.argsort(S)
    S=S[i]
    V=V[:,i]
    V=np.fliplr(V)

  if ws0 and S[0]!=0.:
    print('Warning: forcing S[0] to 0')
    S[0]=0

  # Eigenfunctions (Gn), i.e. the Vertical Structure Functions (Eq. (A8) in Kasahara (1984))
  Gn_all=np.dot(V.T,P_s.T)

  # Re-ordering the eigenfunctions (as k=0,1,...,J-1) and taking the 1st nkMax eigenfunctions:
  nkMax=Plev.size
  Gn=Gn_all[::-1][:nkMax]

  # The equivalent heights re-ordered
  cond=S==0
  S[cond]=1
  hk=const.H00/S
  hk[cond]=np.inf
  hk=hk[:nkMax]

  if save:
    fsave=save_out(dict(Gn=Gn,hk=hk,gamma0=Gamma0,plev=Plev_s, tprof0=Tprof,plev0=Plev),ws0,n_leg,**kargs)
    return Gn[:nk],hk[:nk],fsave
  else: return Gn[:nk],hk[:nk]


def save_out(data,ws0,n_leg,**kargs):
  label=kargs.get('label','out')
  format=kargs.get('format','nc') # nc or npz
  attrs=kargs.get('attrs',{})

  if ws0: ws0='True'
  else:   ws0='False'

  attrs['ws0']           = ws0
  attrs['n_leg']         = np.int8(n_leg)
  import platform
  import sys
  attrs['platform']      = platform.platform()
  attrs['environment']   = 'python'
  attrs['version']       = sys.version
  attrs['version_scipy'] = scipy.__version__
  attrs['version_numpy'] = np.__version__

  fsave='%s_vs_ws0%s.%s'%(label,ws0,format)

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
  nkmax,GL=data['Gn'].shape
  nc.createDimension('nk_max',nkmax)
  nc.createDimension('GL',GL) # Gaussian levels

  # dimensions for original variables:
  nc.createDimension('nlevels0',data['plev0'].size)

  # variables:
  v=nc.createVariable('Gn',data['Gn'].dtype,('nk_max','GL'))
  v.long_name='Vertical structure functions'

  v=nc.createVariable('hk',data['hk'].dtype,('nk_max',))
  v.long_name='Equivalent heights'

  v=nc.createVariable('gamma0',data['gamma0'].dtype,('GL',))
  v.long_name='Static stability in the sigma system'

  v=nc.createVariable('plev',data['plev'].dtype,('GL',))
  v.long_name='Pressure levels corresponding to the gaussian sigma levels'
  v.units='Pa'

  # original variables:
  v=nc.createVariable('plev0',data['plev0'].dtype,('nlevels0',))
  v.long_name='Original pressure levels'
  v.units='Pa'

  v=nc.createVariable('tprof0',data['tprof0'].dtype,('nlevels0',))
  v.long_name='Original temperature profile'
  v.units='K'

  # global attributes:
  import datetime
  nc.date=datetime.datetime.now().isoformat(' ')
  for k in attrs.keys():
    setattr(nc,k,attrs[k])

  # fill variables:
  nc.variables['Gn'][:]=data['Gn']
  nc.variables['hk'][:]=data['hk']
  nc.variables['gamma0'][:]=data['gamma0']
  nc.variables['plev'][:]=data['plev']
  nc.variables['plev0'][:]=data['plev0']
  nc.variables['tprof0'][:]=data['tprof0']

  nc.close()
