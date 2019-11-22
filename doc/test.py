import nmf3d
import numpy as np
import pylab as pl

# ------------------------------------ profile:
f=nmf3d.datafolder+'/T*.nc*'

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
f=nmf3d.datafolder+'/T_ERA_I_1979_2010.txt'
T,Lev=np.loadtxt(f)

GnT,hkT,vfileT=nmf3d.vertical_structure.vse(T,Lev,ws0=True)
GnF,hkF,vfileF=nmf3d.vertical_structure.vse(T,Lev,ws0=False)


# ------------------------------------ Hough:
nk=5
dataT,hfileT=nmf3d.hough_functions.hvf(hkT[:nk],M=6,nLR=8,nLG=6,dlat=6)
dataF,hfileF=nmf3d.hough_functions.hvf(hkF[:nk],M=6,nLR=8,nLG=6,dlat=6)

# ------------------------------------ expansion coeffs (w_nlk)
fu    = nmf3d.datafolder+'/u_01_1979_.nc4'
fv    = nmf3d.datafolder+'/v_01_1979_.nc4'
fz    = nmf3d.datafolder+'/z_01_1979_.nc4'
fzref = nmf3d.datafolder+'/PHI_raw.txt'
from nmf3d import load_ERA_I
data=load_ERA_I.load(fu,fv,fz,fzref,height=False)

wnlkT,efileT=nmf3d.expansion_coeffs.calc(vfileT,hfileT,data,label='outT')
wnlkF,efileF=nmf3d.expansion_coeffs.calc(vfileF,hfileF,data,label='outF')
