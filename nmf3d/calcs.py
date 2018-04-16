from scipy import interpolate
import netCDF4
import numpy as np

isstr=lambda s: isinstance(s,(''.__class__,u''.__class__))

def ncshow(f,**kargs):
  from functools import reduce

  lmax      = kargs.get('lmax',False) # max len of attname of varname
  Lmax      = kargs.get('Lmax',70) # max len of file att or lon_name/units

  if not isstr(f) or any([i in f for i in '*?']):
    nc=netCDF4.MFDataset(f)
  else:
    nc=netCDF4.Dataset(f)

  print('\n# Contents of the NetCDF file')
  print('   '+f)

  print('\n:: Global Attributes:')
  atn=list(nc.ncattrs())
  atv=[getattr(nc,i) for i in atn]
  if atn:
    l1=reduce(max,[len(x) for x in atn])
    try:
      l2=reduce(max,[len(str(x)) for x in atv])
    except:
      l2=reduce(max,[len(unicode(x)) for x in atv])

    if lmax: l1=min(lmax,l1)
    if Lmax: l2=min(Lmax,l2)

    format='   %-'+str(l1) + 's  %-'+str(l2)+'s'
    for i,k in enumerate(atn):
       try: v=str(atv[i])
       except: v=unicode(at[i])
       if len(k)>l1: k=k[:l1-1]+'+'
       if len(v)>l2: v=v[:l2-1]+'+'
       print(format % (k,v))

  print('\n:: Dimensions:')
  din = list(nc.dimensions)
  div=[]
  unlim=[]
  for i in din:
    if hasattr(nc.dimensions[i],'size'):
      div+=[nc.dimensions[i].size]
    else: div+=[nc.dimensions[i].dimtotlen]
    if nc.dimensions[i].isunlimited(): unlim+=[1]
    else: unlim+=[0]

  if din:
    l1=reduce(max,[len(x) for x in din])
    l2=reduce(max,[len(str(x)) for x in div])
    format='   %-'+str(l1) + 's  %'+str(l2)+'d'
    for i,k in enumerate(din):
       if unlim[i]:
         print(format % (k,div[i]) + ' (unlimited)')
       else:
         print(format % (k,div[i]))

  print('\n:: Variables:')
  varnames = list(nc.variables)
  if varnames:
    # find max len
    # for vname:
    l1=reduce(max,[len(x) for x in varnames])
    # for long_name, units and shape:
    l2=14 # min len for long_name
    l3= 7 # min len for units
    l4= 7 # min len for str(shape)
    for v in varnames:
      atn=list(nc.variables[v].ncattrs())
      atv=[getattr(nc.variables[v],i) for i in atn]

      if 'long_name' in atn:
        longn=atv[atn.index('long_name')]
        l2=max(l2,len(longn))
      if 'units' in atn:
        units=atv[atn.index('units')]
        l3=max(l3,len(units))

      l4=max(len(str(nc.variables[v].shape)),l4)

    if Lmax:
      l2=min(l2,Lmax)
      l3=min(l3,Lmax)
      l4=min(l4,Lmax)

    format='   %-'+str(l1)+'s | %-'+str(l2)+'s | %-'+str(l3)+'s | %-'+str(l4)+'s |'
    format1='   %-'+str(l1)+'s   %-'+str(l2)+'s   %-'+str(l3)+'s   %-'+str(l4)+'s'
    print(format1 % ('','long_name'.center(l2),'units'.center(l3),'shape'.center(l4)))
    for v in varnames:
      atn=list(nc.variables[v].ncattrs())
      atv=[getattr(nc.variables[v],i) for i in atn]

      if 'long_name' in atn: longn=atv[atn.index('long_name')]
      else: longn=''
      if len(longn)>l2: longn=longn[:l2-1]+'+'

      if 'units' in atn: units=atv[atn.index('units')]
      else: units=''
      if len(units)>l3: units=units[:l3-1]+'+'

      shape=str(nc.variables[v].shape)
      if len(shape)>l4: shape=shape[:l4-1]+'+'

      print(format % (v,longn,units,shape))

  nc.close()


def leg(n,x,norm=True):
  if n==0:
    if norm:
      return 1/np.sqrt(2)*(1+0*x)
    else:
      return 1+0*x

  rootn = np.sqrt(range(2*n+1))
  s = np.sqrt(1-x**2)
  P = np.zeros((n+3,x.size),x.dtype)

  e=np.geterr()['divide']
  np.seterr(divide='ignore')
  twocot = -2*x/s
  np.seterr(divide=e)

  sn=(-s)**n
  tol=np.finfo(x.dtype).tiny**0.5
  ind = np.where((s>0)& (np.abs(sn)<=tol))[0]
  if ind.size:
    v = 9.2-np.log(tol)/(n*s[ind])
    w = 1/np.log(v)
    m1 = 1+n*s[ind]*v*w*(1.0058+ w*(3.819 - w*12.173))
    m1 = np.minimum(n, np.floor(m1))

    # Column-by-column recursion
    for k in range(len(m1)):
        mm1 = int(m1[k])-1
        col = ind[k]
        P[mm1:n+1,col] = 0

        # Start recursion with proper sign
        tstart = np.finfo(x.dtype).eps
        P[mm1,col] = np.sign(np.remainder(mm1+1,2)-0.5)*tstart
        if x[col] < 0:
            P[mm1,col] = np.sign(np.remainder(n+1,2)-0.5)*tstart

        # Recur from m1 to m = 0, accumulating normalizing factor.
        sumsq = tol
        for m in range(mm1-1,-1,-1):
          P[m+1-1,col] = ((m+1)*twocot[col]*P[m+2-1,col]-
                rootn[n+m+3-1]*rootn[n-m-1]*P[m+3-1,col])/(rootn[n+m+2-1]*rootn[n-m+1-1])

          sumsq = P[m+1-1,col]**2 + sumsq

        scale = 1/np.sqrt(2*sumsq - P[0,col]**2)
        P[:mm1+1,col] = scale*P[:mm1+1,col]

  # Find the values of x,s for which there is no underflow, and for
  # which twocot is not infinite (x~=1).

  nind = np.where((x!=1)&(np.abs(sn)>=tol))[0]

  if nind.size:
    # Produce normalization constant for the m = n function
    c=(1-1/np.arange(2.,2*n+1,2)).prod()

    # Use sn = (-s).^n (written above) to write the m = n function
    P[n,nind] = np.sqrt(c)*sn[nind]
    P[n-1,nind] = P[n,nind]*twocot[nind]*n/rootn[-1]

    # Recur downwards to m = 0
    for m in  range(n-2,-1,-1):
      P[m,nind] = (P[m+1,nind]*twocot[nind]*(m+1) \
            -P[m+2,nind]*rootn[n+m+2]*rootn[n-m-1])/ \
            (rootn[n+m+2-1]*rootn[n-m+1-1])

  y = P[:n+1]

  # Polar argument   (x = +-1)
  s0 = np.where(s==0)[0]
  if s0.size:
    y[0,s0] = x[s0]**n

  if not norm:
    for m in range(n-1):
      y[m+1]=rootn[n-m:n+m+2].prod()*y[m+1]

    y[n] = rootn[1:].prod()*y[n]
  else:
    y = y*(n+0.5)**0.5
    const1 = -1
    for r in range(n+1):
        const1 = -const1
        y[r] = const1*y[r]


  return y


def profile(files,**kargs):
  '''
  verical profile (time/space avg)
  kargs:
    xname, long varname ['longitude']
    yname, lat varname ['latitude']
    zname, vertical varname ['level']
    tname, time varname ['time']
    xmeth, use x data as is ('lin', default) or refine with a spline ('spline')
    zmeth, same as xmeth but for y data
  '''

  debug = kargs.get('debug',0)
  quiet = kargs.get('quiet',0)

  vname = kargs.get('vname',None)
  xname = kargs.get('xname','longitude')
  yname = kargs.get('yname','latitude')
  zname = kargs.get('zname','level')
  tname = kargs.get('tname','time')


  xmeth = kargs.get('xmeth','lin') # along x use data as is ('lin') or refine using 'spline'
  ymeth = kargs.get('ymeth','lin') # same as xmeth but for y

  nc=netCDF4.MFDataset(files)

  #  tdim  = kargs.get('tdim','time')
  #  zdim  = kargs.get('zdim','lev')
  #tdim=nc.variables[tname].dimensions[0]
  #zdim=nc.variables[zname].dimensions[0]

  # look for the main variable name if not provided:
  if not vname:
    vars=[]
    for v in nc.variables:
      if nc.variables[v].ndim==4: vars+=[v]

    if len(vars)==1:
      vname=vars[0]
      if debug>0: print('found 4d var %s'%vname)
    else:
      print('None or more than one 4d var found!!')
      return None,None

  # load coordinates:
  x=nc.variables[xname][:]
  y=nc.variables[yname][:]
  z=nc.variables[zname][:]
  t=nc.variables[tname][:]
  if x.ndim==1: x,y=np.meshgrid(x,y)
  nt=t.size
  nz=z.size

  # start the integration:
  u=np.zeros((nt,nz),'f')
  for k in range(nz):
    if not quiet: print(' - level %d of %d'%(k,nz))
    for it in range(nt):
      #if it%10==0: print('%d of %d'%(it,nt))
      v=nc.variables[vname][it,k]

      # avg for each lat:
      ny,nx=v.shape
      V=np.zeros(ny,v.dtype)
      for j in range(ny):
        x0=x[j]
        v0=v[j]

        if xmeth=='lin':
          # let us assume lons are not cyclic !!
          # but we can check ...
          if np.abs(x0[-1]-x0[0])==360.: v0=v0[:-1]
          V[j]=v0.mean()

        elif xmeth=='spline':
          # order if using spline:
          if x0[-1]<x0[0]: x0,v0=x0[::-1],v0[::-1]

          # make it cyclic
          if x0[-1]-x0[0]!=360.:
            v0=np.hstack((v0,v0[0]))
            if x0[0]==-180: x0=np.hstack((x0,[180.]))
            elif x0[0]==0:  x0=np.hstack((x0,[360.]))

          # interpolate:
          x1=np.linspace(x0[0],x0[-1],nx*5)
          i=interpolate.splrep(x0,v0)
          v1 = interpolate.splev(x1,i)

          # integrate: trapezoidal integration 1/nL*sum(L(Yi+1 + Yi)/2) = Y[:-1].mean()
          V[j]=v1[:-1].mean()
        else:
          print('Unknown xmeth %s: use lin or spline'%xmeth)
          return


      # avg along lat
      y0=y.mean(1) # we probably have just one lat, ie, original lat was 1d

      if ymeth=='lin':
        # avg weighting by cos(lat)
        w=np.cos(y0*np.pi/180.)
        V=(V*w).sum()/w.sum()

      elif ymeth=='spline':
        # order if using spline:
        if y0[-1]<y0[0]: y0,V=y0[::-1],V[::-1]

        y1=np.linspace(y0[0],y0[-1],ny*5)
        i=interpolate.splrep(y0,V)
        V = interpolate.splev(y1,i)

        # avg weighting by cos(lat)
        w=np.cos(y1*np.pi/180.)
        V=(V*w).sum()/w.sum()
      else:
        print('Unknown ymeth %s: use lin or spline'%ymeth)
        return


      u[it,k]=V

  nc.close()

  # simple time avg:
  u=u.mean(0)

  return u,z
