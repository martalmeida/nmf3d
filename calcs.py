from okean import netcdf, calc
import numpy as np


def profile(files,**kargs):
  '''verical profile (time/space avg)'''

  debug = kargs.get('debug',0)

  vname = kargs.get('vname',None)
  xname = kargs.get('xname','lon')
  yname = kargs.get('yname','lat')
  zname = kargs.get('zname','lev')
  tname = kargs.get('tname','time')

  tdim  = kargs.get('tdim','time')
  zdim  = kargs.get('zdim','lev')

  xmeth = kargs.get('xmeth','lin') # along x, use data as is ('lin') or refine using 'splines'
  ymeth = kargs.get('ymeth','lin') # along y: data as is ('lin'), refine using 'splines'
                                   # or convert to 'gauss' points before intergration (not fully
                                   # implemented yet)
  tmeth = kargs.get('tmeth','lin') # I the time dim, use data as is or refine using 'splines'

  nc=netcdf.ncopen(files)

  # look for the main variable name if not provided:
  if not vname:
    vars=[]
    for v in nc.varnames:
      if len(nc.vars[v].dims)==4: vars+=[v]

    if len(vars)==1:
      vname=vars[0]
      if debug>0: print 'found 4d var %s'%vname
    else:
      print 'None or more than one 4d var found!!'
      return None,None

  # load coordinates:
  x=netcdf.use(nc,xname)
  y=netcdf.use(nc,yname)
  z=netcdf.use(nc,zname)
  t=netcdf.use(nc,tname)
  if x.ndim==1: x,y=np.meshgrid(x,y)

  nt=t.size
  nt=1
  nz=z.size

  # start the integration:
  u=np.zeros((nt,nz),'f')
  for k in range(nz):
    print ' - level %d of %d'%(k,nz)
    for it in range(nt):#t.size):
      #if it%10==0: print '%d of %d'%(it,nt)
      v=netcdf.use(nc,vname,**{tdim:it,zdim:k})

      # avg for each lat:
      ny,nx=v.shape
      V=np.zeros(ny,v.dtype)
      for j in range(ny):
        x0=x[j]
        v0=v[j]

        if xmeth=='splines':
          # order if using splines:
          if x0[-1]<x0[0]: x0,v0=x0[::-1],v0[::-1]

          # make it cyclic
          if x0[-1]-x0[0]!=360.:
            v0=np.hstack((v0,v0[0]))
            if x0[0]==-180: x0=np.hstack((x0,[180.]))
            elif x0[0]==0:  x0=np.hstack((x0,[360.]))

          # interpolate:
          x1=np.linspace(x0[0],x0[-1],nx*5)
          i=calc.Interp1_spl(x0,v0)
          v1=i.interp(x1)

          # integrate: trapezoidal integration 1/nL*sum(L(Yi+1 + Yi)/2) = Y[:-1].mean()
          V[j]=v1[:-1].mean()

        else: # linear xmeth
          # let us assume lons are not cyclic !!
          # but we can check ...
          if np.abs(x0[-1]-x0[0])==360.: v0=v0[:-1]
          V[j]=v0.mean()

      # avg along lat
      y0=y.mean(1) # we probably have just one lat, ie, original lat was 1d

      if ymeth in ('splines','gauss'):
        # order if using splines or gaussian weights/points:
        if y0[-1]<y0[0]: y0,V=y0[::-1],V[::-1]

      if ymeth=='lin':
        # avg weighting by cos(lat)
        w=np.cos(y0*np.pi/180.)
        V=(V*w).sum()/w.sum()
      elif ymeth=='gauss':
        print 'Not fully implementyed yet !!'
        return None, None
        y1,w1=calc.leggauss_ab(ny*5,y0[0],y0[-1])
        # interp data at gauss points... let's use splines...
        i=calc.Interp1_spl(y0,V)
        V=i.interp(y1)
        # integrate (avg weighting by cos(lat))
        w=np.cos(y1*np.pi/180.)
        V=(V*w*w1).sum()/(y1[-1]-y1[0])
        V=V/w.sum()

      elif ymeth=='splines':
        i=calc.Interp1_spl(y0,V)
        y1=np.linspace(y0[0],y0[-1],ny*5)
        V=i.interp(y1)
        # avg weighting by cos(lat)
        w=np.cos(y1*np.pi/180.)
        V=(V*w).sum()/w.sum()


      u[it,k]=V

  nc.close()

  # simple time avg:
  u=u.mean(0)

  return u,z


def test_profiles():
  u00,z=profile(files,xmeth='lin',ymeth='lin')
  u01,z=profile(files,xmeth='lin',ymeth='splines')  # ok
  u10,z=profile(files,xmeth='splines',ymeth='lin') # ok
  u11,z=profile(files,xmeth='splines',ymeth='splines')

#  u02,z=profile(files,xmeth='lin',ymeth='gauss')    # dif
#
#
#  np.savez('tests.npz',u00=u00,u01=u01,u02=u02,u10=u10,z=z)
#
#  u11,z=profile(files,xmeth='splines',ymeth='splines')
#  u12,z=profile(files,xmeth='splines',ymeth='gauss')

# u00, u01, u02, u10

  import pylab as pl
  pl.plot(u00,z)
  pl.plot(u01,z)
  pl.plot(u02,z)

  pl.plot(u10,z)
  pl.plot(u11,z)
  pl.plot(u12,z)

