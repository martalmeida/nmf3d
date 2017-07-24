import numpy as np
from . import calcs
from . import constants as const

dType='d'
cType=np.complex128

def hvf_bar(nLR,nLG,M,trunc,x):
  '''
  Compute the Hough vector functions as described in the paper of Swarztrauber and Kasahara (1985).
  This is for the limiting case of hk[0] = inf (The Haurwitz waves)

  Part I: The frequencies and the Hough functions are computed for zonal wave number m = 0.
  Part II: The frequencies and the Hough functions are computed for zonal wave numbers m > 0.

  See hough_functions
  '''

  print('- HVF barotropic -')

  N=trunc # truncation order
  NEH=1
  L = nLR+2*nLG  # Total number of meridional modes used in the expansion (should be even)

  # Total number of eigenvalues/eigenvectors for matrices A and B.
  maxN = 3*N

  # Dimensions --------------------
  # Arrays for the Hough functions
  HOUGH_0_UVZ = np.zeros((3,L,len(x)),dtype=dType) # Hough functions for n=0 (for projection studies)
  HOUGH_0_UVZ_2rec   = np.zeros((3,L,len(x)),dtype=dType) # Hough functions for n=0 (for reconstitutions studies)


  print('Part I')
  # PART I ----------------------------------------------------------
  # For zonal wave number n=0.

  # HOUGH VECTOR FUNCTIONS ============================================
  #  Normalized Associated Legendre Functions (Pm_n) ------------------
  # Computes the associated Legendre functions of degree N and order M = 0, 1, ..., N,
  # evaluated for each element of X. N must be a scalar integer and X must contain real values between
  #  -1 <= X <= 1.


  # P(1,n)
  P1_n = np.zeros((2*N,x.size),dtype=dType)
  #P1_n[0,:]=0 # n=0 => P(1,0)=0

  for n in range(1,2*N):
    P1_n[n,:] = calcs.leg(n,x,True)[1] # P(1,n)

  #   P(0,n+1)
  P0_nM1 = np.zeros((2*N,x.size),dtype=dType)
  for n in range(2*N):
    P0_nM1[n] = calcs.leg(n+1,x,True)[0]   # P(0,n+1)

  # P(0,n-1)
  P0_nm1 = np.zeros((2*N,x.size),dtype=dType)
  #P0_nm1[0,:] = 0   # n=0 => P(0,-1)=0
  #P0_nm1[1,:] = 0   # n=1 => P(0,0)=Const=0. We set the constant to zero because we are
                     # dealing with geopotential perturbations (i.e. with zero mean)

  for n in range(2,2*N):
     P0_nm1[n] = calcs.leg(n-1,x,True)[0]

  p0_n=np.zeros(2*N,dtype=dType)
  p0_n1=np.zeros(2*N,dtype=dType)
  for n in range(1,2*N+1):
    # From eq. (3.11) in Swarztrauber and Kasahara (1985).
    p0_n[n-1]  = np.sqrt( ((n+1)) / (n*(2.*n-1)*(2*n+1)) )       # P(0,n)/sqrt(n(n-1))
    p0_n1[n-1] = np.sqrt( (    n) / ((n+1)*(2.*n+1)*(2*n+3) ))   # P(0,n+1)/sqrt((n+1)(n+2))

  # Replicate to have dimensions (2*N x nLat)
  p0_nMAT  = np.tile(p0_n,(x.size,1)).T
  p0_n1MAT  = np.tile(p0_n1,(x.size,1)).T


  # HOUGH vector functions -------------------------------------------
  # The HOUGH vector functions are computed using eq. (3.22) in Swarztrauber and Kasahara (1985)

  # Rotational (ROSSBY) MODES
  HOUGH_0_UVZ[0,2*nLG+1-1:] = - P1_n[1:nLR+1]   # Eq. (5.1)

  # Eq. (5.13)
  HOUGH_0_UVZ_2rec[1-1,2*nLG+1-1:] =  - P1_n[1:nLR+1]

  HOUGH_0_UVZ_2rec[3-1,2*nLG+1-1:] =  (2*const.Er*const.Om)/np.sqrt(const.g) * (p0_nMAT[:nLR]* P0_nm1[1:nLR+1] + p0_n1MAT[:nLR]*P0_nM1[1:nLR+1])
  # Note: The third component of "HOUGH_0_UVZ_2rec" was multiplied by sqrt(g) in order to use the same algorithm in the reconstruction as
  # that used with dimensionalised variables, by setting artificially the barotropic equivalent height (which is infinity) to one.

  # GRAVITY MODES
  # These modes are all zero

  print('End of part I (zonal wave number zero)')


  print('Part II')
  # For zonal wave numbers m>0. The frequencies and associated horizontal stucture functions are
  # computed as the eigenvectors/eigenvalues of matrices A and B in Swarztrauber and Kasahara (1985).
  # Matrices A and B have dimensions 3*N x 3*N because the frequencies are determined in triplets
  # corresponding to eastward gravity, westward gravity and westward rotational (rossby) modes.

  # Dimensions --------------------
  # Arrays for the Hough functions
  HOUGH_UVZ   = np.zeros((3,M,L,x.size),dtype=cType)    # Hough functions for m>0 (for projection studies)
  HOUGH_UVZ_2rec = np.zeros((3,M,L,x.size),dtype=cType) # Hough functions for m>0 (for reconstitutions studies)

  for m in range(1,M+1):  # Start the zonal wave numbers
    # HOUGH VECTOR FUNCTIONS
    # The normalized Legendre functions (Pm_n)

    Pm_n    = np.zeros((2*N,x.size),dtype=dType)
    Pmm1_n  = np.zeros((2*N,x.size),dtype=dType)
    PmM1_n  = np.zeros((2*N,x.size),dtype=dType)
    Pmm1_n1 = np.zeros((2*N,x.size),dtype=dType)
    PmM1_n1 = np.zeros((2*N,x.size),dtype=dType)
    Pm_nm1  = np.zeros((2*N,x.size),dtype=dType)
    Pm_nM1  = np.zeros((2*N,x.size),dtype=dType)

    for n in range(m,2*N-1+m+1):
      Pm_n[n+1-m-1] = calcs.leg(n,x,True)[m+1-1] # P(m,n)
      Pmm1_n[n+1-m-1] = calcs.leg(n,x,True)[m-1] # P(m-1,n)

      if n<m+1:
        PmM1_n[n+1-m-1] = 0
      else:
        PmM1_n[n+1-m-1] = calcs.leg(n,x,True)[m+2-1] # P(m+1,n)

      Pmm1_n1[n+1-m-1]=calcs.leg(n-1,x,True)[m-1] # P(m-1,n-1)

      if n-1<m+1:
        PmM1_n1[n+1-m-1] = 0
      else:
        PmM1_n1[n+1-m-1] =calcs.leg(n-1,x,True)[m+2-1] # P(m+1,n-1)

      if n-1<m:
        Pm_nm1[n+1-m-1] = 0
      else:
        Pm_nm1[n+1-m-1]=calcs.leg(n-1,x,True)[m+1-m] # P(m,n-1)

      Pm_nM1[n+1-m-1] =calcs.leg(n+1,x,True)[m+1-1] # P(m,n+1)


    # Derivative of associated Legendre functions with respect to latitude (eq. (3.3))
    dPm_n_dLat=np.zeros((2*N,x.size),dtype=dType)
    for n in range(m,2*N-1+m+1):
      dPm_n_dLat[n+1-m-1] = 0.5*(np.sqrt((n-m)*(n+m+1))*PmM1_n[n+1-m-1]-np.sqrt((n+m)*(n-m+1))*Pmm1_n[n+1-m-1])


    # The term given by eq. (3.4)
    mPm_n_cosLat=np.zeros((2*N,x.size),dtype=dType)
    for n in range(m,2*N-1+m+1):
      mPm_n_cosLat[n+1-m-1] = 0.5*np.sqrt((2*n+1)/(2*n-1.))*(np.sqrt((n+m)*(n+m-1))*Pmm1_n1[n+1-m-1]+np.sqrt((n-m)*(n-m-1))*PmM1_n1[n+1-m-1])


    # The spherical vector harmonics
    # The spherical vector harmonics will be computed using eqs. (3.1) without the factor e^(i m lambda), since
    # this factor will be canceled in the summation (3.22).
    y2m_n = np.zeros((3,2*N,x.size),dtype=cType)
    y1m_n = np.zeros((3,2*N,x.size),dtype=cType)

    y3m_nm1 = np.zeros((3,2*N,x.size),dtype=cType)
    y3m_nM1 = np.zeros((3,2*N,x.size),dtype=cType)

    y3m_nm1[2] = Pm_nm1
    y3m_nM1[2] = Pm_nM1
    for n in range(m,2*N-1+m+1):
      y1m_n[0,n+1-m-1] = const.I*mPm_n_cosLat[n+1-m-1]/np.sqrt(n*(n+1))
      y1m_n[1,n+1-m-1] = dPm_n_dLat[n+1-m-1]/np.sqrt(n*(n+1))

    y2m_n[0] = -y1m_n[1]
    y2m_n[1] = y1m_n[0]

    pm_n  = np.zeros((2*N,x.size),dtype=dType)
    pm_n1 = np.zeros((2*N,x.size),dtype=dType)
    for n in range(m,2*N-1+m+1):
       # From eq. (3.11) in Swarztrauber and Kasahara (1985).
       pm_n[n+1-m-1]  = np.sqrt( ((n+1)*(n-m)*(n+m)) / (n**3*(2*n-1.)*(2*n+1)) )    # P(m,n)/sqrt(n(n-1))
       pm_n1[n+1-m-1] = np.sqrt( (n*(n-m+1)*(n+m+1))/((n+1)**3*(2*n+1.)*(2*n+3) ))  # P(m,n+1)/sqrt((n+1)(n+2))

    # HOUGH vector functions -----------------------------------------
    # The HOUGH vector functions are computed using eq. (3.22) in Swarztrauber and Kasahara (1985)

    # Rotational (ROSSBY) MODES --------------------------------------
    HOUGH_UVZ[0,m-1,2*nLG+1-1:] = y2m_n[0,:nLR]
    HOUGH_UVZ[1,m-1,2*nLG+1-1:] = y2m_n[1,:nLR]

    HOUGH_UVZ_2rec[0,m-1,2*nLG+1-1:] = y2m_n[0,:nLR]
    HOUGH_UVZ_2rec[1,m-1,2*nLG+1-1:] = y2m_n[1,:nLR]
    HOUGH_UVZ_2rec[2,m-1,2*nLG+1-1:] = (2*const.Er*const.Om)/np.sqrt(const.g) * (pm_n[:nLR]*y3m_nm1[2,:nLR] + pm_n1[:nLR]*y3m_nM1[2,:nLR])
    # Note: The third component of "HOUGH_UVZ_2rec" was multiplied by sqrt(g) in order to use the same algorithm in the reconstruction as
    # that used with dimensionalised variables, by setting artificially the barotropic equivalent height (which is infinity) to one.

    # GRAVITY MODES
    # These modes are all zero

  # End the zonal wave numbers

  # Arrays for the frequencies (eigenvalues)
  SIGMAS = np.zeros((M,nLR),dtype=dType)
  S_auxS = np.zeros((M,maxN),dtype=dType)

  for m in range(1,M+1):
    for n in range(m,maxN+1):
      S_auxS[m-1,n-1] = -m/(n*(n+1.))

    SIGMAS[m-1] = S_auxS[m-1,m-1:nLR+m-1]

  print('End of part II (zonal wave numbers m>0)')


  out=dict(HOUGH_UVZ=HOUGH_UVZ,HOUGH_0_UVZ=HOUGH_0_UVZ,SIGMAS=SIGMAS,
           HOUGH_UVZ_2rec=HOUGH_UVZ_2rec,HOUGH_0_UVZ_2rec=HOUGH_0_UVZ_2rec)

  return out
