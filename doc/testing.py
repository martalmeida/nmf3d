import nmf3d
import numpy as np
import pylab as pl

datafolder='./nmf3d_data/'

# ------------------------------------ profile:
f=datafolder+'/T*.nc*'

import os, glob
files=glob.glob(f)
files.sort()
for f in files:
  print('%s %.1fMb'%(os.path.basename(f),os.stat(f).st_size/1024**2))

t,lev_mb=nmf3d.calcs.profile(files,quiet=1)
lev=lev_mb*100 # Pa
pl.plot(t,lev)
pl.ylabel('Pressure (Pa)')
pl.xlabel('Temperature (K)')
pl.title('Reference temperature profile')
pl.gca().invert_yaxis()

# ------------------------------------ vertical structure:
f=datafolder+'/T_ERA_I_1979_2010.txt'
T,Lev=np.loadtxt(f)

GnT,hkT,vfileT=nmf3d.vertical_structure.vse(T,Lev,ws0=True)
GnF,hkF,vfileF=nmf3d.vertical_structure.vse(T,Lev,ws0=False)


# ------------------------------------ Hough functions:
M=6
nLR=8
nLG=6
nk=5 # number of function to keep

dataT,hfileT=nmf3d.hough_functions.hvf(hkT[:nk],M,nLR,nLG,dlat=6)
dataF,hfileF=nmf3d.hough_functions.hvf(hkF[:nk],M,nLR,nLG,dlat=6)

# ------------------------------------ expansion coefficients:
# 3-D spectrum of total energy W_nlk
fu    = datafolder+'/u_01_1985_.nc4'
fv    = datafolder+'/v_01_1985_.nc4'
fz    = datafolder+'/z_01_1985_.nc4'
fzref = datafolder+'/PHI_raw.txt'

from nmf3d import load_ERA_I
data=load_ERA_I.load(fu,fv,fz,fzref,height=False)

w_nlkT,wfileT=nmf3d.expansion_coeffs.calc(vfileT,hfileT,data,label='outT')
w_nlkF,wfileF=nmf3d.expansion_coeffs.calc(vfileF,hfileF,data,label='outF')

# energy
w_nlk=w_nlkT
nk,nM,nL,nT=w_nlk.shape
E0=np.zeros((nk,nL,nT))
En=np.zeros((nk,nM-1,nL,nT))

for i in range(nk):
    E0[i]=1/4*1e5*hkT[i]*(w_nlk[i,0]*w_nlk[i,0].conj()).real
    En[i]=1/2*1e5*hkT[i]*(w_nlk[i,1:]*np.conj(w_nlk[i,1:])).real

# 3-D spectrum of energy interactions
if 0:
  # assuming the coefficients were calculated and stored in the files
  # I1.npy, I2.npy and J3.npy; and the user has the variables lon, lat and P:

  data_i1=dict(lon=lon,lat=lat,P=P,v=np.load('I1.npy'))
  data_i2=dict(lon=lon,lat=lat,P=P,v=np.load('I2.npy'))
  data_j3=dict(lon=lon,lat=lat,P=P,v=np.load('J3.npy'))

  data_i=dict(I1=data_i1,I2=data_i2)
  i_nlk,ifsave=nmf3d.expansion_coeffs.calc(vfileF,hfileF,data_i,label='out_i_ws0_True')

  idata_j=dict(J3=data_j3)
  j_nlkF,jfsave=nmf3d.expansion_coeffs.calc(vfileF,hfileF,data_j,label='out_j_ws0_True')

# storing the vertical transform
if 0:
  u=data.u.v
  u_k,fsave=nmf3d.transforms.vertical(u,hk,nk,Gn,p_old,p_new,dataLabel,'meridional wind')

# ------------------------------------ inverse expansion coefficients:
zi=[1,2,3]
mi=[1,2,3]
vi=[1,2]
pl=[850,500]
lon=np.arange(0,360,30)

uvz,invfsave=nmf3d.inv_expansion_coeffs.calc(vfileT,hfileT,wfileT,zi,mi,vi,pl,lon)
