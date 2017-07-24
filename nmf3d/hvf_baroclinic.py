import numpy as np
from numpy.lib import scimath
import scipy as sp
from . import constants as const

dType='d'
cType=np.complex128

def hvf(hk,M,nLR,nLG,latType,dlat):
  '''
  Compute the Hough vector functions as described in the paper of Swarztrauber and Kasahara (1985).
  Baroclinic mode. hk[0] can't be inf

  Part I: The frequencies and the Hough functions are computed for zonal wave number m = 0.
  Part II: The frequencies and the Hough functions are computed for zonal wave numbers m > 0.

  See hough_functions
  '''

  print('- HVF baroclinic -')

  L = nLR + 2*nLG  # Total number of meridional modes used in the expansion (should be even)

  # The equivalent heights (the first nk)
  NEH=hk.size


  # Dimensionless constant (gamma) computed from (2.8) of Swarztrauber and Kasahara (1985):
  # gamma=sqrt(g*hk)/(2*Er*Om), where hk are the equivalent heights obtained as the
  # solution of the Vertical Structure Equations
  Ga = np.sqrt(const.g*hk) / (2*const.Er*const.Om)

  # Lamb's parameter (epson)
  Ep = Ga**-2

  # Truncation order for the expansion in terms of the spherical vector harmonics
  N = np.max([20,L,np.ceil(np.sqrt(Ep).max()).astype('i')]) # why 20? see Swarztrauber and A. Kasahara (1985), pg 481.
  truncation_order=N # returned cos is needed for the barotropic if ws0 is True

  # Total number of eigenvalues/eigenvectors for matrices A and B.
  maxN = 3*N

  # Latitude points in a 1.5 degree grid (ERA-Interim)
  if latType=='linear':
    LAT = np.arange(-90.0,90.0+dlat/2.,dlat)
    x   = np.sin(LAT*np.pi/180.)
  elif latType=='gaussian':
    x,w=np.polynomial.legendre.leggauss(N)


  print('Part I')
  # PART I ----------------------------------------------------------
  # For zonal wave number n=0, the frequencies and associated horizontal
  # stucture functions are computed as the eigenvalues/eigenvectors of
  # matrices C, D, E and F given in Swarztrauber and Kasahara (1985).
  # Since the frequencies are not determined in triplets, matrices
  # C, D, E and F have dimensions N x N.

  # Matrices C and D =================================================

  # p_n, Eq. (4.2) in Swarztrauber and Kasahara (1985)
  p_n = np.zeros(2*N,dtype=dType)
  for k in range(2*N):
    p_n[k] = np.sqrt((k*(k+2.0))/((2*k+1)*(2*k+3)))

  # r_n, Eq. (3.26) in Swarztrauber and Kasahara (1985)
  r_n = np.zeros((2*N,NEH),dtype=dType)
  for gi in range(NEH):
    for k in range(2*N):
      r_n[k,gi] = Ga[gi] * np.sqrt(k*(k+1))

  print('  - Matrix C')
  C,S_C,U_C = calc_CD(p_n,r_n,0)

  print('  - Matrix D')
  D,S_D,U_D = calc_CD(p_n,r_n,1)

  # Matrices E and F =================================================

  # Term  n(n+1) of eq. (4.19) in Swarztrauber and Kasahara (1985)
  n_n1 = np.zeros((2*N+1,1)) # int
  for n in np.arange(0,2*N+1):
    n_n1[n] = (n+1) * (n+2)


  # Terms d_n and e_n of eqs. (4.18) in Swarztrauber and Kasahara (1985)
  d_n = np.zeros((2*N+2,NEH),dtype=cType)
  e_n = np.zeros((2*N+2,NEH),dtype=dType)
  for gi in range(NEH):
     d_n[:,gi] = (np.arange(2*N+2)-1) / (Ga[gi]*scimath.sqrt((2*np.arange(2*N+2)-1)*(2*np.arange(2*N+2)+1)))
     e_n[:,gi] = (np.arange(2*N+2)+2) / (Ga[gi]*np.sqrt((2*np.arange(2*N+2)+1)*(2*np.arange(2*N+2)+3)))


  # Matrix E is given by eq. (4.21) in Swarztrauber and Kasahara (1985)
  print('  - Matrix E')

  E_dd = np.zeros((N,NEH),dtype=cType)
  E_md = np.zeros((N+1,NEH),dtype=dType)

  # Upper (and lower) diagonal elements
  for k in range(N):
    E_dd[k,:] = d_n[2*k,:] * e_n[2*k,:];

  # d_n[0,:] is purely imaginary. Therefore d_n[0,:]*d_n[0,:] is real.
  # Using np.real(d_n[0,:]*d_n[0,:]) avoids python ComplexWarning ("Casting complex
  # values to real discards the imaginary part")
  E_md[0,:] = np.real(d_n[0,:]*d_n[0,:])
  for k in range(1,N+1):
    # Main diagonal elements for Matrix E.
    E_md[k,:] = n_n1[2*k-2] + e_n[2*k-2,:] * e_n[2*k-2,:] + np.real(d_n[2*k,:]*d_n[2*k,:])

  # Creating the tridiagonal matrix E and calculating it's eigenvalues/eigenvectors
  # Creating matrix diag(i,1,...,1) for diagonal similarity transform of matrix E
  AUX_E      = np.eye((N+1),dtype=cType)
  AUX_E[0,0] = scimath.sqrt(-1)

  E       = np.zeros((N+1,N+1,NEH),dtype=cType)
  U_E     = np.zeros((N+1,N+1,NEH),dtype=cType)
  U_Eaux  = np.zeros((N+1,N+1,NEH),dtype=dType)
  AUXe    = np.zeros((N+1,N+1,NEH),dtype=dType)
  Sa_E    = np.zeros((N+1,NEH)    ,dtype=dType)
  S_Eaux  = np.zeros((N+1,NEH)    ,dtype=dType)
  IE2sort = np.zeros((N+1,NEH),'i')

  for gi in range(NEH):
     # Tridiagonal matrix E
     E[:,:,gi] = np.diag(E_dd[:,gi],k=-1) + np.diag(E_md[:,gi]) + np.diag(E_dd[:,gi],k=1)

     # The eigenvalues/eigenvectors of E [Eq. (4.22) in Swarztrauber and Kasahara (1985)]
     AUXe[:,:,gi],U_Eaux[:,:,gi] = np.linalg.eig(np.real(np.dot(np.dot(np.conjugate(AUX_E).T, np.squeeze(E[:,:,gi])), AUX_E)))
     U_E[:,:,gi]                 = np.dot(AUX_E,np.squeeze(U_Eaux[:,:,gi]))   # Multliplying eigenvectors by diag(i,1,...,1)
     S_Eaux[:,gi]                = np.diag(np.squeeze(AUXe[:,:,gi]))          # The eigenvalues unsorted
     IE2sort[:,gi]               = np.argsort(S_Eaux[:,gi],axis=0)            # Returns the indices that would sort the eigenvalues.
     S_Eaux[:,gi]                = S_Eaux[IE2sort[:,gi],gi]                   # Sorting the eigenvalues.
     U_E[:,:,gi]                 = U_E[:,IE2sort[:,gi],gi]                    # Sorting the corresponding eigenvectors.


  # Frequencies Sigma_a, Eqs. (4.18) in Swarztrauber and Kasahara (1985)
  Sa_E = -1.0 / S_Eaux


  # Matrix F is given by eq. (4.24) in Swarztrauber and Kasahara (1985)
  print('  - Matrix F')

  F_dd = np.zeros((N-1,NEH),dtype=dType)
  F_md = np.zeros((N,NEH),dtype=dType)

  for k in range(1,N):
    # Upper (and lower) diagonal elements for Matrix F
    F_dd[k-1,:] = np.real(d_n[2*k+1,:]) * e_n[2*k+1,:]

  for k in range(N):
    # Main diagonal elements for Matrix F.
    F_md[k,:] =  n_n1[2*k+1] + e_n[2*k+1,:] * e_n[2*k+1,:] + np.real(d_n[2*k+3,:] * d_n[2*k+3,:])

  # Matrix F, eigenvalues/eigenvectors
  F       = np.zeros((N,N,NEH),dtype=dType)
  U_F     = np.zeros((N,N,NEH),dtype=dType)
  AUXf    = np.zeros((N,N,NEH),dtype=dType)
  Sa_F    = np.zeros((N,NEH)  ,dtype=dType)
  S_Faux  = np.zeros((N,NEH)  ,dtype=dType)
  IF2sort = np.zeros((N,NEH),'i')

  for gi in range(NEH):
    # Tridiagonal matrix F
    F[:,:,gi] = np.diag(F_dd[:,gi],k=-1) + np.diag(F_md[:,gi]) + np.diag(F_dd[:,gi],k=1)
    # The eigenvalues/eigenvectors of F [Eq. (4.25) in Swarztrauber and Kasahara (1985)]
    AUXf[:,:,gi],U_F[:,:,gi] = np.linalg.eigh(np.squeeze(F[:,:,gi]))
    S_Faux[:,gi]             = np.diag(np.squeeze(AUXf[:,:,gi]));      # the eigenvalues unsorted
    IF2sort[:,gi]            = np.argsort(abs(S_Faux[:,gi]),axis=0);   # Returns the indices that would sort the eigenvalues.
    S_Faux[:,gi]             = S_Faux[IF2sort[:,gi],gi];               # Sorting the eigenvalues.
    U_F[:,:,gi]              = U_F[:,IF2sort[:,gi],gi];                # Sorting the corresponding eigenvectors.

  # Frequencies Sigma_a, Eqs. (4.18) in Swarztrauber and Kasahara (1985)
  Sa_F = -1.0 / S_Faux


  # Selecting the frequencies and the coefficients for the vector harmonic expansion
  # The eigenvalues are classified as:
  # Westward gravity    => from eigenvalues of matrices C and D
  # Westward Rotational => from eigenvalues of matrices E and F
  # Eastward gravity    => from eigenvalues of matrices C and D

  # Coefficients An, Bn and Cn =======================================
  print('  - Coeffs An, Bn, Cn')

  # Case 1 : Gravity Modes
  #
  # Coefficients An are obtained from eigenvectors of matrices C and D, and
  # coefficients Bn and Cn are obtained from An using eqs. (4.4) and (4.5)

  # Arrays for the frequencies (eigenvalues)
  WEST_G_0_sy  = np.zeros((nLG//2,NEH),dtype='d');
  WEST_G_0_asy = np.zeros((nLG//2,NEH),dtype='d');
  EAST_G_0_sy  = np.zeros((nLG//2,NEH),dtype='d');
  EAST_G_0_asy = np.zeros((nLG//2,NEH),dtype='d');

  # Arrays for the coefficients (eigenvectors)
  # The coefficients are stored in columns as:
  # symmetric subsystem     --> (C0, A0, B1, C2, A2, B3, ...)
  # antisymmetric subsystem --> (B0, C1, A1, B2, C3, A3, ...)
  ABC_WG_0_sy  = np.zeros((maxN,nLG//2,NEH),dtype=dType)
  ABC_WG_0_asy = np.zeros((maxN,nLG//2,NEH),dtype=dType)
  ABC_EG_0_sy  = np.zeros((maxN,nLG//2,NEH),dtype=dType)
  ABC_EG_0_asy = np.zeros((maxN,nLG//2,NEH),dtype=dType)

  An_G_0_sy    = np.zeros((N,nLG//2,NEH),dtype=dType)
  An_G_0_asy   = np.zeros((N,nLG//2,NEH),dtype=dType)
  Bn_G_0_sy    = np.zeros((N,nLG//2,NEH),dtype=dType)
  Bn_G_0_asy   = np.zeros((N,nLG//2,NEH),dtype=dType)
  Cn_G_0_sy    = np.zeros((N,nLG//2,NEH),dtype=dType)
  Cn_G_0_asy   = np.zeros((N,nLG//2,NEH),dtype=dType)

  # Frequencies (eigenvalues)
  WEST_G_0_sy  = -S_C[:nLG//2,:]
  WEST_G_0_asy = -S_D[:nLG//2,:]
  #
  EAST_G_0_sy  =  S_C[:nLG//2,:]
  EAST_G_0_asy =  S_D[:nLG//2,:]

  # Coefficients An (eigenvectors)
  An_G_0_sy  = U_C[:,:nLG//2,:]  # symmetric subsystem -> [A0, A2, A4, ... , A2N-2] <- from matrix C
  An_G_0_asy = U_D[:,:nLG//2,:]  # antisymmetric subsystem -> [A1, A3, A5, ... , A2N-1] <- from matrix D


  # Computation of Bn and Cn
  # Cn => Eq. (4.5) in Swarztrauber and Kasahara (1985)
  r_n_aux = np.zeros((2*N,nLG//2,NEH),dtype=dType)
  for s in range(nLG//2):
    r_n_aux[:,s,:] = r_n;

  for n in range(N):
    for l in range(1,nLG//2):
      # This loop starts at l=1 to avoid division by zero warning (i.e. EAST_G_0_sy[l=0,:]=0).
      # Variables ABC_WG_0_sy and ABC_EG_0_sy, which dependend on Cn_G_0_sy, are changed below 
      # (cf. p.475 of Swarztrauber and Kasahara (1985)) because eastward gravity mode of lowest
      # order (l=0) is eigenvector of matrix E.

      # symmetric subsystem -> (C0, C2, C4, ... , C2N-2) <- from matrix C
      Cn_G_0_sy[n,l,:] = r_n_aux[2*n,l,:].T * An_G_0_sy[n,l,:].T / EAST_G_0_sy[l,:]

  for n in range(N):
    for l in range(nLG//2):
      # antisymmetric subsystem -> (C1, C3, C5, ... , C2N-1) <- from matrix D
      Cn_G_0_asy[n,l,:] = r_n_aux[2*n+1,l,:].T * An_G_0_asy[n,l,:].T / EAST_G_0_asy[l,:]

  # Bn => Eq. (4.4) in Swarztrauber and Kasahara (1985)
  # Last Bn for the symmetric subsystem must be computed from eq. (4.3)
  for n in np.arange(0,N-1):
    for l in np.arange(1,nLG//2):
      # This loop starts at l=1. See above.

      # symmetric subsystem -> (B1, B3, B5, ... ) <- from matrix C
      Bn_G_0_sy[n,l,:] = (p_n[2*n]*An_G_0_sy[n,l,:].T+p_n[2*n+1]*An_G_0_sy[n+1,l,:].T) / EAST_G_0_sy[l,:]

  # Last Bn for the symmetric subsystem computed from eq. (4.3)
  for l in range(nLG//2):
    Bn_G_0_sy[N-1,l,:] = (EAST_G_0_sy[l,:]*An_G_0_sy[N-1,l,:].T-r_n[2*N-2,:]*Cn_G_0_sy[N-1,l,:].T-p_n[2*N-3]*Bn_G_0_sy[N-1,l,:].T) / p_n[2*N-2]


  Bn_G_0_asy[0,:,:] = 0; # B0 = 0 from (4.4)
  for n in range(1,N):
    for l in range(nLG//2):
      # antisymmetric subsystem -> (B0, B2, B4, ... ) <- from matrix D
      Bn_G_0_asy[n,l,:] = (p_n[2*n-1]*An_G_0_asy[n-1,l,:].T+p_n[2*n]*An_G_0_asy[n,l,:].T) / EAST_G_0_asy[l,:]


  # Storing the coefficients in columns as:
  # symmetric subsystem     --> (C0, A0, B1, C2, A2, B3, ...)
  # antisymmetric subsystem --> (B0, C1, A1, B2, C3, A3, ...)

  n1 = 0;
  for n in np.arange(0,maxN,3):
    # Eastward gravity:
    # - symmetric subsystem:
    ABC_EG_0_sy[n,:,:]    = Cn_G_0_sy[n1,:,:]
    ABC_EG_0_sy[n+1,:,:]  = An_G_0_sy[n1,:,:]
    ABC_EG_0_sy[n+2,:,:]  = Bn_G_0_sy[n1,:,:]
    # - antisymmetric subsystem:
    ABC_EG_0_asy[n,:,:]   = Bn_G_0_asy[n1,:,:]
    ABC_EG_0_asy[n+1,:,:] = Cn_G_0_asy[n1,:,:]
    ABC_EG_0_asy[n+2,:,:] = An_G_0_asy[n1,:,:]
    #
    # Westward gravity:
    # - symmetric subsystem:
    ABC_WG_0_sy[n,:,:]    = Cn_G_0_sy[n1,:,:]
    ABC_WG_0_sy[n+1,:,:]  = -An_G_0_sy[n1,:,:]
    ABC_WG_0_sy[n+2,:,:]  = Bn_G_0_sy[n1,:,:]
    # - antisymmetric subsystem:
    ABC_WG_0_asy[n,:,:]   = Bn_G_0_asy[n1,:,:]
    ABC_WG_0_asy[n+1,:,:] = Cn_G_0_asy[n1,:,:]
    ABC_WG_0_asy[n+2,:,:] = -An_G_0_asy[n1,:,:]
    #
    n1=n1+1;


  # There are two modes with zero frequency which are classified as gravity modes.
  # The first is obtained by noting that U=(1,0,0,...)^T is an eigenvector of C,
  # which corresponds to eigenvalue sigma^2=0. Therefore the first westward gravity
  # mode is identified with A0=1 and all other coeficients are zero.
  ABC_WG_0_sy[:,0,:] = 0  # all other coeficients are zero.
  ABC_WG_0_sy[1,0,:] = 1  # A0=1 (cf. p.475 of Swarztrauber and Kasahara (1985)).


  # Case 2 : Rossby(rotational) Modes
  #
  # Coefficients An are all zero
  # Coefficients Bn_til are obtained from eigenvectors of matrices E and F
  # Bn and Cn are obtained from Bn_til using eqs. (4.18) and (4.26)
  # (4.18) -> Bn=sqrt(n(n+1))*Bn_til
  # (4.26) -> Cn=-dn*Bn-1_til - en*Bn+1_til.

  # Frequencies (eigenvalues)
  WEST_R_0_sy  = Sa_E[:nLR//2]
  WEST_R_0_asy = Sa_F[:nLR//2]

  # Arrays for the coefficients (eigenvectors)
  # The coefficients are stored in columns as:
  # symmetric subsystem     --> (C0, A0, B1, C2, A2, B3, ...)
  # antisymmetric subsystem --> (B0, C1, A1, B2, C3, A3, ...)
  ABC_WR_0_sy  = np.zeros((maxN,nLR//2+1,NEH),dtype=dType)
  An_R_0_sy    = np.zeros((N+1,nLR//2+1,NEH) ,dtype=dType)
  auxBn_R_0_sy = np.zeros((N+1,nLR//2+1,NEH) ,dtype=dType)
  Bn_R_0_sy    = np.zeros((N+1,nLR//2+1,NEH) ,dtype=dType)
  Cn_R_0_sy    = np.zeros((N+1,nLR//2+1,NEH) ,dtype=dType)
  Bn_til_0_sy  = np.zeros((N+1,nLR//2+1,NEH) ,dtype=cType)

  ABC_WR_0_asy  = np.zeros((maxN,nLR//2,NEH),dtype=dType)
  An_R_0_asy    = np.zeros((N,nLR//2,NEH)   ,dtype=dType)
  auxBn_R_0_asy = np.zeros((N,nLR//2,NEH)   ,dtype=dType)
  Bn_R_0_asy    = np.zeros((N,nLR//2,NEH)   ,dtype=dType)
  Cn_R_0_asy    = np.zeros((N,nLR//2,NEH)   ,dtype=dType)
  Bn_til_0_asy  = np.zeros((N,nLR//2,NEH)   ,dtype=dType)

  for la in range(nLR//2):
    # The frequencies (eigenvalues)
    WEST_R_0_sy[la,:]  = Sa_E[la,:]
    WEST_R_0_asy[la,:] = Sa_F[la,:]
    # The coefficients Bn_til [eigenvectors]
    Bn_til_0_sy[:,la,:]  = U_E[:,la,:]   #     symmetric subsystem --> [Bn_til[-1], Bn_til[1], Bn_til[3], ... ] <-- matrix E
    Bn_til_0_asy[:,la,:] = U_F[:,la,:]   # antisymmetric subsystem --> [ Bn_til[2], Bn_til[4], Bn_til[6], ... ] <-- matrix F

  Bn_til_0_sy[:,nLR//2,:] = U_E[:,nLR//2,:]


  # Computation of Bn and Cn
  aux_sy = np.zeros((N+1),dtype=dType)
  nn1 = 0
  for n in range(-1,2*N,2):
    aux_sy[nn1] = np.sqrt(n*(n+1))
    nn1 = nn1 + 1

  aux_asy = np.zeros((N),dtype=dType)
  nn2 = 0
  for n in range(2,2*N+2,2):
    aux_asy[nn2] = np.sqrt(n*(n+1))
    nn2 = nn2 + 1

  # symmetric subsystem --> (B-1, B1, B3, B5 ... ) <-- from matrix E (with N+1 elements)
  for n in range(N+1):   # Element N+1 of Bn_til_0_sy is used to compute last element of Bn_R_0_sy
    # Eq. (4.18) in Swarztrauber and Kasahara (1985)
    auxBn_R_0_sy[n,:,:] = np.real(aux_sy[n]*Bn_til_0_sy[n,:,:])

  # The Bn's to be stored are (B1, B3, B5 ... ), hence:
  Bn_R_0_sy[0:N,:,:] = auxBn_R_0_sy[1:N+1,:,:]


  # symmetric subsystem --> (C0, C2, C4, ... ) <-- from matrix E (with N+1 elements)
  for n in range(N):   # Only the first N elements will be used
    for k in range(NEH):
      # Eq. (4.26) in Swarztrauber and Kasahara (1985)
      Cn_R_0_sy[n,:,k] = np.real(-d_n[2*n,k]*Bn_til_0_sy[n,:,k] - e_n[2*n,k]*Bn_til_0_sy[n+1,:,k])


  for n in range(1,N):
    for k in np.arange(0,NEH):
      # antisymmetric subsystem --> (B2, B4, B6, ... ) <-- from matrix F (with N elements)
      auxBn_R_0_asy[n,:,k] = aux_asy[n] * Bn_til_0_asy[n,:,k]
      Cn_R_0_asy[n,:,k]    = np.real(-d_n[2*n+1,k]*Bn_til_0_asy[n-1,:,k]-e_n[2*n+1,k]*Bn_til_0_asy[n,:,k])

  n1 = 0
  for k in range(NEH):
    # antisymmetric subsystem --> (B2, B4, B6, ... ) <-- from matrix F (with N elements)
    auxBn_R_0_asy[n1,:,k] = aux_asy[n1]    * Bn_til_0_asy[n1,:,k]
    Cn_R_0_asy[n1,:,k]    = -e_n[2*n1+1,k] * Bn_til_0_asy[n1,:,k]

  ## OR
  #for n in np.arange(0,N):
    #for k in np.arange(0,NEH):
      ## antisymmetric subsystem --> (B2, B4, B6, ... ) <-- from matrix F (with N elements)
      #auxBn_R_0_asy[n,:,k] = aux_asy[n]*Bn_til_0_asy[n,:,k];
      #if n==0:
        #Cn_R_0_asy[n,:,k]  = -e_n[2*n+1,k]*Bn_til_0_asy[n,:,k];
      #else:
        #Cn_R_0_asy[n,:,k]  = -d_n[2*n+1,k]*Bn_til_0_asy[n-1,:,k] - e_n[2*n+1,k]*Bn_til_0_asy[n,:,k];


  # The Bn's to be stored are (B0=0, B2, B4 ... ), hence:
  Bn_R_0_asy[1:N-1,:,:] = auxBn_R_0_asy[0:N-2,:,:]
  Bn_R_0_asy[0,:,:]     = 0


  # Storing the coefficients in columns as:
  # symmetric subsystem     --> (C0, A0, B1, C2, A2, B3, ...)
  # antisymmetric subsystem --> (B0, C1, A1, B2, C3, A3, ...)
  n1 = 0
  for n in np.arange(0,maxN,3):
    # _____________Westward rotational_______________
    # ------------ symmetric subsystem ------------ #
    ABC_WR_0_sy[n,:,:]    = Cn_R_0_sy[n1,:,:]
    # An coefficients are all zero
    ABC_WR_0_sy[n+2,:,:]  = Bn_R_0_sy[n1,:,:]
    # ---------- antisymmetric subsystem ---------- #
    ABC_WR_0_asy[n,:,:]   = Bn_R_0_asy[n1,:,:]
    ABC_WR_0_asy[n+1,:,:] = Cn_R_0_asy[n1,:,:]
    # An coefficients are all zero
    # ____________________________________________
    n1 = n1 + 1

  #  The eastward gravity mode of lowest order
  #  is eigenvector of matrix E. Therefore,
  ABC_EG_0_sy[:,0,:] = ABC_WR_0_sy[:,0,:]


  # Normalising the coefficients -------------------------------------
  NormaWG_sy  = np.zeros((nLG//2,NEH)  ,dtype=dType)
  NormaWG_asy = np.zeros((nLG//2,NEH)  ,dtype=dType)
  NormaEG_sy  = np.zeros((nLG//2,NEH)  ,dtype=dType)
  NormaEG_asy = np.zeros((nLG//2,NEH)  ,dtype=dType)
  NormaWR_sy  = np.zeros((nLR//2+1,NEH),dtype=dType)
  NormaWR_asy = np.zeros((nLR//2,NEH)  ,dtype=dType)

  for la in np.arange(0,nLG//2):
    for nh in np.arange(0,NEH):
      NormaWG_sy[la,nh]  = np.linalg.norm(ABC_WG_0_sy[:,la,nh])
      NormaWG_asy[la,nh] = np.linalg.norm(ABC_WG_0_asy[:,la,nh])
      NormaEG_sy[la,nh]  = np.linalg.norm(ABC_EG_0_sy[:,la,nh])
      NormaEG_asy[la,nh] = np.linalg.norm(ABC_EG_0_asy[:,la,nh])

  for la in np.arange(0,nLR//2):
    for nh in np.arange(0,NEH):
      NormaWR_sy[la,nh]  = np.linalg.norm(ABC_WR_0_sy[:,la,nh])
      NormaWR_asy[la,nh] = np.linalg.norm(ABC_WR_0_asy[:,la,nh])

  for nh in np.arange(0,NEH):
    NormaWR_sy[nLR//2,nh]  = np.linalg.norm(ABC_WR_0_sy[:,nLR//2,nh])

  ABC_WG_0_sy  = ABC_WG_0_sy/NormaWG_sy
  ABC_WG_0_asy = ABC_WG_0_asy/NormaWG_asy
  ABC_EG_0_sy  = ABC_EG_0_sy/NormaEG_sy
  ABC_EG_0_asy = ABC_EG_0_asy/NormaEG_asy
  ABC_WR_0_sy  = ABC_WR_0_sy/NormaWR_sy
  ABC_WR_0_asy = ABC_WR_0_asy/NormaWR_asy


  # Because the westward gravity modes have been obtained as minus the square root of the
  # eigenvalues of C and D, their order may be reversed (optionally).
  #  for n in np.arange(0,maxN):
  #     ABC_WG_0_sy[n,:,:]  = fliplr(squeeze(ABC_WG_0_sy[n,:,:]));
  #     ABC_WG_0_asy[n,:,:] = fliplr(squeeze(ABC_WG_0_asy[n,:,:]));


  # Storing the symmetric and antisymmetric coefficients for the Rossby modes in one array
  ABC_WR_0 = np.zeros((maxN,nLR,NEH),dtype=dType)

  # Similarly to the first westward gravity mode (which is identified with A0=1 and all
  # other coeficients zero), the first westward rotational mode is identified with B0=1
  #  and all other coeficients zero.
  ABC_WR_0[0,0,:] = 1;   # B0=1 and all other coeficients are zero.

  for l in range(1,nLR//2):
    ABC_WR_0[:,2*l-1,:] = ABC_WR_0_sy[:,l,:]
    ABC_WR_0[:,2*l,:]   = ABC_WR_0_asy[:,l-1,:]


  ABC_WR_0[:,nLR-1,:]   = ABC_WR_0_sy[:,nLR//2,:]


  # HOUGH VECTOR FUNCTIONS ============================================
  #  Normalized Associated Legendre Functions (Pm_n) ------------------
  #  The associated Legendre functions are evaluated for degrees n=0,...,2*N and orders M = 0 and 1, for
  #  each element of X. N must be a scalar integer and X must contain real values between -1 <= x <= 1.

  # M=0 ---------------------------
  print('  - Normalized Associated Legendre Functions - M=0')
  # Deffining the array for the Associated Legendre Functions (Unnormalized and normalized)
  uP0_n = np.zeros((2*N,x.size),dtype=dType) # unnormalized
  P0_n  = np.zeros((2*N,x.size),dtype=dType) # normalized

  # The order
  m = 0

  # Computing first two Associated Legendre Functions
  for n in range(2):
    # Looping over the latitudes in x
    for z in range(x.size):
      P0n,dP0n_dz = sp.special.lpmn(m, n, x[z])
      uP0_n[n,z]    = P0n[m,n]
      n_m           = np.math.factorial(n-m)
      nm            = np.math.factorial(n+m)
      P0_n[n,z]     = (-1)**m * np.sqrt((n+0.5)*n_m/(nm)) * uP0_n[n,z]; # normalizing

  # Computing the remaining Associated Legendre Functions by recurrence relation (accurate)
  for n in range(2,2*N):
    P0_n[n,:] = np.sqrt((4*(n**2) -1)/(n**2-m**2)) * (x*P0_n[n-1,:]-np.sqrt(((n-1)**2-m**2)/(4*(n-1)**2 -1))*P0_n[n-2,:])

  # Alternative (slower)
  ## Looping over first half of the degrees (0:N)
  #for n in np.arange(0,N):
  ## Looping over the latitudes in x
    #for z in np.arange(0,x.size):
      #[P0n,dP0n_dz] = sp.special.lpmn(m, n, x[z])
      #uP0_n[n,z]    = P0n[m,n];
      #n_m           = sp.factorial(n-m);
      #nm            = sp.factorial(n+m);
      #P0_n[n,z]     = (-1)**m*np.sqrt((n+0.5)*n_m/(nm))*uP0_n[n,z]; # normalizing

  ## Looping over the second half of the degrees (N+1:2*N), and
  ## computing Legendre Functions by recurrence relation
  #for n in np.arange(N,2*N):
    #P0_n[n,:] = np.sqrt( (4*(n**2) -1) / (n**2 - m**2) ) * ( x*P0_n[n-1,:] - np.sqrt( ((n-1)**2 - m**2) / (4*(n-1)**2 -1) )*P0_n[n-2,:] )


  # M=1 ---------------------------
  print('  - Normalized Associated Legendre Functions - M=1')
  # Deffining the array for the Associated Legendre Functions (Unnormalized and normalized)
  uP1_n = np.zeros((2*N,x.size),dtype=dType)   # unnormalized
  P1_n  = np.zeros((2*N,x.size),dtype=dType)   # normalized

  # The order
  m = 1

  # m must be <= n, therefore P1_0=0 for m=1 and n=0.
  #P1_n[0,:] = 0

  # Computing first two Associated Legendre Functions
  for n in range(1,3):
    # Looping over the latitudes in x
    for z in np.arange(0,x.size):
      P1n,dP1n_dz = sp.special.lpmn(m, n, x[z])
      uP1_n[n,z]  = P1n[m,n]
      n_m         = np.math.factorial(n-m)
      nm          = np.math.factorial(n+m)
      P1_n[n,z]   = (-1)**m * np.sqrt((n+0.5)*n_m/(nm)) * uP1_n[n,z]   # normalizing

  # Computing the remaing Associated Legendre Functions by recurrence relation
  for n in range(3,2*N):
    P1_n[n,:] = np.sqrt((4*(n**2)-1)/(n**2-m**2)) * (x*P1_n[n-1,:]-np.sqrt(((n-1)**2-m**2)/(4*(n-1)**2 -1))*P1_n[n-2,:])


  # Dimensions --------------------
  # Arrays for the Hough functions
  HOUGH_0_UVZ = np.zeros((3,L,NEH,len(x)),dtype=cType)  # Hough functions for n=0.
  AUX_0_UVZ   = np.zeros((3,L,NEH,len(x)),dtype=cType)  # Auxiliar array for summation (3.22)

  # HOUGH vector functions --------
  # computed using eq. (3.22) in Swarztrauber and Kasahara (1985)
  #
  # GRAVITY MODES
  print('  - HVF: gravity modes')

  try:
    from progressbar import ProgressBar
    pBar=ProgressBar(maxval=nLG//2+1-1,term_width=50)
    pBar.start()
  except: pBar=False

  I=const.I
  for l in range(1,nLG//2+1):
    #print('     l=%02d:%d'%(l,nLG//2+1))
    if pBar: pBar.update(l-1)
    for neh in range(1,NEH+1):
      for n in range(1,int(N)+1):
        # ========================== SYMMETRIC SUBSISTEMS ======================
        # --------------------------- Westward gravity -------------------------
        # ----------------------------- Component 1 ----------------------------
        AUX_0_UVZ[0,2*l-2,neh-1,:]   = - ABC_WG_0_sy[3*n-1,l-1,neh-1] * P1_n[2*n-1,:] + 0*I
        HOUGH_0_UVZ[0,2*l-2,neh-1,:] = HOUGH_0_UVZ[0,2*l-2,neh-1,:] + AUX_0_UVZ[0,2*l-2,neh-1,:] + 0*I
        # ----------------------------- Component 2 ----------------------------
        AUX_0_UVZ[1,2*l-2,neh-1,:]   = I*ABC_WG_0_sy[3*n-2,l-1,neh-1] * P1_n[2*n-2,:]
        HOUGH_0_UVZ[1,2*l-2,neh-1,:] = HOUGH_0_UVZ[1,2*l-2,neh-1,:] + AUX_0_UVZ[1,2*l-2,neh-1,:]
        # ----------------------------- Component 3 ----------------------------
        AUX_0_UVZ[2,2*l-2,neh-1,:]   = -ABC_WG_0_sy[3*n-3,l-1,neh-1] * P0_n[2*n-2,:] + 0*I
        HOUGH_0_UVZ[2,2*l-2,neh-1,:] = HOUGH_0_UVZ[2,2*l-2,neh-1,:] + AUX_0_UVZ[2,2*l-2,neh-1,:] + 0*I
        # --------------------------- Eastward gravity -------------------------
        # ---------------------------- Component 1 -----------------------------
        AUX_0_UVZ[0,2*l-2+nLG,neh-1,:]   = - ABC_EG_0_sy[3*n-1,l-1,neh-1] * P1_n[2*n-1,:] + 0*I
        HOUGH_0_UVZ[0,2*l-2+nLG,neh-1,:] = HOUGH_0_UVZ[0,2*l-2+nLG,neh-1,:] + AUX_0_UVZ[0,2*l-2+nLG,neh-1,:] + 0*I
        # ----------------------------- Component 2 ----------------------------
        AUX_0_UVZ[1,2*l-2+nLG,neh-1,:]   = I*ABC_EG_0_sy[3*n-2,l-1,neh-1] * P1_n[2*n-2,:]
        HOUGH_0_UVZ[1,2*l-2+nLG,neh-1,:] = HOUGH_0_UVZ[1,2*l-2+nLG,neh-1,:] + AUX_0_UVZ[1,2*l-2+nLG,neh-1,:]
        # ----------------------------- Component 3 ----------------------------
        AUX_0_UVZ[2,2*l-2+nLG,neh-1,:]   = -ABC_EG_0_sy[3*n-3,l-1,neh-1] * P0_n[2*n-2,:] + 0*I
        HOUGH_0_UVZ[2,2*l-2+nLG,neh-1,:] = HOUGH_0_UVZ[2,2*l-2+nLG,neh-1,:] + AUX_0_UVZ[2,2*l-2+nLG,neh-1,:] + 0*I
        # ========================== ANTISYMMETRIC SUBSISTEMS ==================
        # --------------------------- Westward gravity -------------------------
        # ---------------------------- Component 1 -----------------------------
        AUX_0_UVZ[0,2*l-1,neh-1,:]   = - ABC_WG_0_asy[3*n-3,l-1,neh-1] * P1_n[2*n-2,:] + 0*I
        HOUGH_0_UVZ[0,2*l-1,neh-1,:] = HOUGH_0_UVZ[0,2*l-1,neh-1,:] + AUX_0_UVZ[0,2*l-1,neh-1,:] + 0*I
        # ---------------------------- Component 2 -----------------------------
        AUX_0_UVZ[1,2*l-1,neh-1,:]   = I*ABC_WG_0_asy[3*n-1,l-1,neh-1] * P1_n[2*n-1,:]
        HOUGH_0_UVZ[1,2*l-1,neh-1,:] = HOUGH_0_UVZ[1,2*l-1,neh-1,:] + AUX_0_UVZ[1,2*l-1,neh-1,:]
        # ---------------------------- Component 3 -----------------------------
        AUX_0_UVZ[2,2*l-1,neh-1,:]   = -ABC_WG_0_asy[3*n-2,l-1,neh-1] * P0_n[2*n-1,:] + 0*I
        HOUGH_0_UVZ[2,2*l-1,neh-1,:] = HOUGH_0_UVZ[2,2*l-1,neh-1,:] + AUX_0_UVZ[2,2*l-1,neh-1,:] + 0*I
        # --------------------------- Eastward gravity -------------------------
        # ---------------------------- Component 1 -----------------------------
        AUX_0_UVZ[0,2*l-1+nLG,neh-1,:]   = - ABC_EG_0_asy[3*n-3,l-1,neh-1] * P1_n[2*n-2,:] + 0*I
        HOUGH_0_UVZ[0,2*l-1+nLG,neh-1,:] = HOUGH_0_UVZ[0,2*l-1+nLG,neh-1,:] + AUX_0_UVZ[0,2*l-1+nLG,neh-1,:] + 0*I
        # ---------------------------- Component 2 -----------------------------
        AUX_0_UVZ[1,2*l-1+nLG,neh-1,:]   = I*ABC_EG_0_asy[3*n-1,l-1,neh-1] * P1_n[2*n-1,:]
        HOUGH_0_UVZ[1,2*l-1+nLG,neh-1,:] = HOUGH_0_UVZ[1,2*l-1+nLG,neh-1,:] + AUX_0_UVZ[1,2*l-1+nLG,neh-1,:]
        # ---------------------------- Component 3 -----------------------------
        AUX_0_UVZ[2,2*l-1+nLG,neh-1,:]   = -ABC_EG_0_asy[3*n-2,l-1,neh-1] * P0_n[2*n-1,:] + 0*I
        HOUGH_0_UVZ[2,2*l-1+nLG,neh-1,:] = HOUGH_0_UVZ[2,2*l-1+nLG,neh-1,:] + AUX_0_UVZ[2,2*l-1+nLG,neh-1,:] + 0*I

  if pBar: pBar.finish()

  #
  # ROSSBY MODES
  print('  - HVF: rossby modes')

  if pBar:
    pBar=ProgressBar(maxval=nLR//2+1-1,term_width=50)
    pBar.start()

  for l in range(1,nLR//2+1):
    if pBar: pBar.update(l-1)
    for neh in range(1,NEH+1):
      for n in range(1,int(N)+1):
        # ================================= SYMMETRIC SUBSISTEMS ================================
        # --------------------------------- Westward rossby -------------------------------------
        # ----------------------------------- Component 1 ---------------------------------------
        AUX_0_UVZ[0,2*l-1+2*nLG,neh-1,:]   = - ABC_WR_0[3*n-1,2*l-1,neh-1] * P1_n[2*n-1,:] + 0*I;
        HOUGH_0_UVZ[0,2*l-1+2*nLG,neh-1,:] = HOUGH_0_UVZ[0,2*l-1+2*nLG,neh-1,:] + AUX_0_UVZ[0,2*l-1+2*nLG,neh-1,:] + 0*I;
        # ----------------------------------- Component 2 ---------------------------------------
        AUX_0_UVZ[1,2*l-1+2*nLG,neh-1,:]   = I*ABC_WR_0[3*n-2,2*l-1,neh-1] * P1_n[2*n-2,:];
        HOUGH_0_UVZ[1,2*l-1+2*nLG,neh-1,:] = HOUGH_0_UVZ[1,2*l-1+2*nLG,neh-1,:] + AUX_0_UVZ[1,2*l-1+2*nLG,neh-1,:];
        # ----------------------------------- Component 3 ---------------------------------------
        AUX_0_UVZ[2,2*l-1+2*nLG,neh-1,:]   = -ABC_WR_0[3*n-3,2*l-1,neh-1] * P0_n[2*n-2,:];
        HOUGH_0_UVZ[2,2*l-1+2*nLG,neh-1,:] = HOUGH_0_UVZ[2,2*l-1+2*nLG,neh-1,:] + AUX_0_UVZ[2,2*l-1+2*nLG,neh-1,:];
        # ================================= ANTISYMMETRIC SUBSISTEMS ===============================
        # --------------------------------- Westward rossby -------------------------------------
        # ----------------------------------- Component 1 ---------------------------------------
        AUX_0_UVZ[0,2*l-2+2*nLG,neh-1,:]   = - ABC_WR_0[3*n-3,2*l-2,neh-1] * P1_n[2*n-2,:] + 0*I;
        HOUGH_0_UVZ[0,2*l-2+2*nLG,neh-1,:] = HOUGH_0_UVZ[0,2*l-2+2*nLG,neh-1,:] + AUX_0_UVZ[0,2*l-2+2*nLG,neh-1,:] + 0*I;
        # ------------------------------------ Component 2 ---------------------------------------
        AUX_0_UVZ[1,2*l-2+2*nLG,neh-1,:]   = I*ABC_WR_0[3*n-1,2*l-2,neh-1] * P1_n[2*n-1,:];
        HOUGH_0_UVZ[1,2*l-2+2*nLG,neh-1,:] = HOUGH_0_UVZ[1,2*l-2+2*nLG,neh-1,:] + AUX_0_UVZ[1,2*l-2+2*nLG,neh-1,:];
        # ----------------------------------- Component 3 ---------------------------------------
        AUX_0_UVZ[2,2*l-2+2*nLG,neh-1,:]   = -ABC_WR_0[3*n-2,2*l-2,neh-1] * P0_n[2*n-1,:] + 0*I;
        HOUGH_0_UVZ[2,2*l-2+2*nLG,neh-1,:] = HOUGH_0_UVZ[2,2*l-2+2*nLG,neh-1,:] + AUX_0_UVZ[2,2*l-2+2*nLG,neh-1,:] + 0*I;

  if pBar: pBar.finish()

  print('End of part I (zonal wave number zero)')

  print('Part II')
  # PART II ------------------------------------------------------------
  # For zonal wave numbers m>0. The frequencies and associated horizontal stucture functions are
  # computed as the eigenvalues/eigenvectors of matrices A and B in Swarztrauber and Kasahara (1985).
  # Matrices A and B have dimensions 3*N x 3*N because the frequencies are determined in triplets
  # corresponding to eastward gravity, westward gravity and westward rotational (rossby) modes.


  # Dimensions --------------------
  # Arrays for the Hough functions
  HOUGH_UVZ   = np.zeros((3,M,L,NEH,len(x)),dtype=cType)   # Hough functions for m>0.
  AUX_UVZ     = np.zeros((3,M,L,NEH,len(x)),dtype=cType)   # Auxiliar array for summation (3.22)

  # Arrays for the frequencies (eigenvalues)
  WEST_G_sy  = np.zeros((M,nLG//2,NEH),dtype=dType)
  WEST_G_asy = np.zeros((M,nLG//2,NEH),dtype=dType)
  EAST_G_sy  = np.zeros((M,nLG//2,NEH),dtype=dType)
  EAST_G_asy = np.zeros((M,nLG//2,NEH),dtype=dType)

  WEST_R_sy  = np.zeros((M,nLR//2,NEH),dtype=dType)
  WEST_R_asy = np.zeros((M,nLR//2,NEH),dtype=dType)

  # Arrays for the coefficients (eigenvectors)
  # The coefficients are stored in columns as:
  # Symmetric subsystem     ---> (Cm_m,  Am_m,  Bm_m+1, Cm_m+2, Am_m+2, Bm_m+3, ...)
  # Antisymmetric subsystem ---> (Bm_m, Cm_m+1, Am_m+1, Bm_m+2, Cm_m+3, Am_m+3, ...)
  ABC_WG_sy  = np.zeros((M,maxN,nLG//2,NEH),dtype=dType)
  ABC_WG_asy = np.zeros((M,maxN,nLG//2,NEH),dtype=dType)
  ABC_EG_sy  = np.zeros((M,maxN,nLG//2,NEH),dtype=dType)
  ABC_EG_asy = np.zeros((M,maxN,nLG//2,NEH),dtype=dType)

  ABC_WR_sy  = np.zeros((M,maxN,nLR//2,NEH),dtype=dType)
  ABC_WR_asy = np.zeros((M,maxN,nLR//2,NEH),dtype=dType)


  # calculations
  # Matrices A (symmetric subsystem) and B (antisymmetric subsystem)

  for m in range(1,M+1):
    print('  %d of %d'%(m,M))

    # Terms r_n, qm_n and pm_n
    n=m+np.arange(0,maxN)
    r_n=np.zeros((maxN,NEH),dtype=dType)
    for gl in np.arange(1,NEH+1):
      for k in np.arange(1,maxN+1):
        # Eq. (3.26) in Swarztrauber and Kasahara (1985)
        r_n[k-1,gl-1] = Ga[gl-1] * np.sqrt(n[k-1]*(n[k-1]+1))

    pm_n = np.zeros((maxN),dtype=dType)
    qm_n = np.zeros((maxN),dtype=dType)

    for k in range(1,(maxN)+1):
      # Eq. (3.11) in Swarztrauber and Kasahara (1985).
      pm_n[k-1] = np.sqrt(((n[k-1]-1)*(n[k-1]+1)*(n[k-1]-m)*(n[k-1]+m))/(n[k-1]**2*(2*n[k-1]-1)*(2*n[k-1]+1)))
      # Eq. (3.11) in Swarztrauber and Kasahara (1985)
      qm_n[k-1] = m / (n[k-1]*(n[k-1]+1))


    # Matrix A is given by eq. (3.28) in Swarztrauber and Kasahara (1985)
    if m==1: print('  - Matrix A')
    #-----------------------------------------------------------------
    A_uud = np.zeros((maxN,NEH),dtype=dType)   # Uppermost diagonal elements for Matrix A.
    A_ud  = np.zeros((maxN,NEH),dtype=dType)   # Upper diagonal elements for Matrix A.
    A_md  = np.zeros((maxN,NEH),dtype=dType)   # Main diagonal elements for Matrix A.
    for k in range(1,maxN//3+1):
      A_uud[3*k-1] =  pm_n[2*k]
      A_ud[3*k-3]  =  r_n[2*k-2]
      A_ud[3*k-2]  =  pm_n[2*k-1]
      A_md[3*k-2]  = -qm_n[2*k-2]
      A_md[3*k-1]  = -qm_n[2*k-1]

    A_uud = A_uud[0:maxN-2]   # Only the first maxN-2 elements are needed.
    A_ud  = A_ud[0:maxN-1]    # Only the first maxN-1 elements are needed.
    #A_lld=A_uud.copy();         # Lowermost diagonal elements for Matrix A (not needed).
    #A_ld=A_ud.copy();           # Lower diagonal elements for Matrix A (not needed).

    # Creating the pentadiagonal matrix A and calculating it's eigenvalues/eigenvectors.
    A    = np.zeros((maxN,maxN,NEH),dtype=dType)
    S_A  = np.zeros((maxN,NEH)     ,dtype=dType)
    U_A  = np.zeros((maxN,maxN,NEH),dtype=dType)
    IA2sort = np.zeros((maxN,NEH)  ,dtype=int)

    for gl in range(1,NEH+1):
      # Pentadiagonal matrix A.
      A[:,:,gl-1]      = np.diag(A_uud[:,gl-1],k=-2) + np.diag(A_ud[:,gl-1],k=-1) + np.diag(A_md[:,gl-1],k=0) + np.diag(A_ud[:,gl-1],k=1) + np.diag(A_uud[:,gl-1],k=2)
      # The eigenvalues/eigenvectors of A [eq. (3.29) in Swarztrauber and Kasahara (1985)]
      [auxAUXa,auxU_A] = np.linalg.eigh(np.squeeze(A[:,:,gl-1]))
      S_A[:,gl-1]      = auxAUXa                            # The eigenvalues.
      U_A[:,:,gl-1]    = auxU_A                             # The corresponding eigenvectors.
      IA2sort[:,gl-1]  = np.argsort(S_A[:,gl-1],axis=0)     # Returns the indices that would sort the eigenvalues.
      S_A[:,gl-1]      = S_A[IA2sort[:,gl-1],gl-1]          # Sorting the eigenvalues.
      U_A[:,:,gl-1]    = U_A[:,IA2sort[:,gl-1],gl-1]        # Sorting the corresponding eigenvectors.


    # Matrix B is given by eq. (3.31) in Swarztrauber and Kasahara (1985)
    if m==1: print('  - Matrix B')
    #-----------------------------------------------------------------
    B_uud = np.zeros((maxN,NEH),dtype=dType)   # Uppermost diagonal elements for Matrix B.
    B_ud  = np.zeros((maxN,NEH),dtype=dType)   # Upper diagonal elements for Matrix B.
    B_md  = np.zeros((maxN,NEH),dtype=dType)   # Main diagonal elements for Matrix B.
    for k in range(1,maxN//3+1):
      B_uud[3*k-3] =  pm_n[2*k-1]
      B_ud[3*k-2]  =  r_n[2*k-1]
      B_ud[3*k-1]  =  pm_n[2*k]
      B_md[3*k-3]  = -qm_n[2*k-2]
      B_md[3*k-1]  = -qm_n[2*k-1]

    B_uud = B_uud[0:maxN-2]   # Only the first maxN-2 elements are needed.
    B_ud  = B_ud[0:maxN-1]    # Only the first maxN-1 elements are needed.
    #B_lld=B_uud.copy()         # Lowermost diagonal elements for Matrix B (not needed).
    #B_ld=B_ud.copy()           # Lower diagonal elements for Matrix B (not needed).


    # Creating the pentadiagonal matrix A and calculating it's eigenvalues/eigenvectors.
    B       = np.zeros((maxN,maxN,NEH),dtype=dType)
    S_B     = np.zeros((maxN,NEH)     ,dtype=dType)
    U_B     = np.zeros((maxN,maxN,NEH),dtype=dType)
    IB2sort = np.zeros((maxN,NEH)     ,dtype=int)

    for gl in range(1,NEH+1):
      # Pentadiagonal matrix A.
      B[:,:,gl-1]      = np.diag(B_uud[:,gl-1],k=-2) + np.diag(B_ud[:,gl-1],k=-1) + np.diag(B_md[:,gl-1],k=0) + np.diag(B_ud[:,gl-1],k=1) + np.diag(B_uud[:,gl-1],k=2)
      # The eigenvalues/eigenvectors of B [eq. (3.32) in Swarztrauber and Kasahara (1985)]
      [auxAUXb,auxU_B] = np.linalg.eigh(np.squeeze(B[:,:,gl-1]))
      S_B[:,gl-1]      = auxAUXb                            # The eigenvalues.
      U_B[:,:,gl-1]    = auxU_B                             # The corresponding eigenvectors.
      IB2sort[:,gl-1]  = np.argsort(S_B[:,gl-1],axis=0)     # Returns the indices that would sort the eigenvalues.
      S_B[:,gl-1]      = S_B[IB2sort[:,gl-1],gl-1]          # Sorting the eigenvalues.
      U_B[:,:,gl-1]    = U_B[:,IB2sort[:,gl-1],gl-1]        # Sorting the corresponding eigenvectors.


    if m==1: print('  - selecting freqs and coeffs')
    # Selecting the frequencies and the coefficients for the vector harmonic expansion.
    # The eigenvalues are classified (for both symmetric and antisymmetric subsystems) as:
    # Westward Gravity -----> (lowest third)
    # Westward Rotational --> (middle third)
    # Eastward Gravity -----> (highest third).

    # Gravity modes
    for la in range(1,nLG//2+1):
      # The frequencies (eigenvalues)
      WEST_G_sy[m-1,la-1,:]    = S_A[N+1-la-1,:]
      WEST_G_asy[m-1,la-1,:]   = S_B[N+1-la-1,:]
      EAST_G_sy[m-1,la-1,:]    = S_A[2*N+la-1,:]
      EAST_G_asy[m-1,la-1,:]   = S_B[2*N+la-1,:]
      # The coefficients A, B and C (eigenvectors)
      ABC_WG_sy[m-1,:,la-1,:]  = U_A[:,N+1-la-1,:]
      ABC_WG_asy[m-1,:,la-1,:] = U_B[:,N+1-la-1,:]
      ABC_EG_sy[m-1,:,la-1,:]  = U_A[:,2*N+la-1,:]
      ABC_EG_asy[m-1,:,la-1,:] = U_B[:,2*N+la-1,:]

    # Rossby modes
    for la in np.arange(1,nLR//2+1):
      # The frequencies (eigenvalues)
      WEST_R_asy[m-1,la-1,:]   = S_B[N+la-1,:]
      WEST_R_sy[m-1,la-1,:]    = S_A[N+la-1,:]
      # The coefficients A, B and C (eigenvectors)
      ABC_WR_asy[m-1,:,la-1,:] = U_B[:,N+la-1,:]
      ABC_WR_sy[m-1,:,la-1,:]  = U_A[:,N+la-1,:]


    # HOUGH VECTOR FUNCTIONS ===========================================
    #  Normalized Associated Legendre functions (Pm_n) -----------------
    #  The associated Legendre functions are evaluated for degrees n=0,...,2*N and orders M = 0 and 1,for
    #  each element of x. N must be a scalar integer and X must contain real values between -1 <= x <= 1.

    # Deffining the array for the Associated Legendre functions (Unnormalized and normalized)
    # The Unnormalized
    uPm_n    = np.zeros((2*N,x.size),dtype=dType)
    uPmm1_n  = np.zeros((2*N,x.size),dtype=dType)
    uPmM1_n  = np.zeros((2*N,x.size),dtype=dType)
    uPmm1_n1 = np.zeros((2*N,x.size),dtype=dType)
    uPmM1_n1 = np.zeros((2*N,x.size),dtype=dType)
    # The Normalized
    Pm_n    = np.zeros((2*N,x.size),dtype=dType)
    Pmm1_n  = np.zeros((2*N,x.size),dtype=dType)
    PmM1_n  = np.zeros((2*N,x.size),dtype=dType)
    Pmm1_n1 = np.zeros((2*N,x.size),dtype=dType)
    PmM1_n1 = np.zeros((2*N,x.size),dtype=dType)


    if m==1: print('  - Associated Legendre Functions')
    # Computing first two Associated Legendre Functions
    for n in range(m,m+2):
      # Looping over the latitudes in x
      for z in range(x.size):
        # P(m,n)
        Pmn,dPmn_dz  = sp.special.lpmn(m, n, x[z])
        uPm_n[n-m,z] = Pmn[m,n]
        n_m          = np.math.factorial(n-m)
        nm           = np.math.factorial(n+m)
        Pm_n[n-m,z]  = (-1)**m * np.sqrt((n+0.5)*n_m/(nm)) * uPm_n[n-m,z] # P(m,n) normalized

        # P(m-1,n)
        Pmm1n,dPmm1n_dz = sp.special.lpmn(m-1, n, x[z])
        uPmm1_n[n-m,z]  = Pmm1n[m-1,n]
        n_m             = np.math.factorial(n-(m-1))
        nm              = np.math.factorial(n+(m-1))
        Pmm1_n[n-m,z]   = (-1)**(m-1) * np.sqrt((n+0.5)*n_m/(nm)) * uPmm1_n[n-m,z] # P(m-1,n) normalized

        # P(m+1,n)
        PmM1n,dPmM1n_dz  = sp.special.lpmn(m+1, n+1, x[z])
        uPmM1_n[n-m+1,z] = PmM1n[m+1,n+1]
        n_m              = np.math.factorial(n+1-(m+1))
        nm               = np.math.factorial(n+1+(m+1))
        PmM1_n[n-m+1,z]  = (-1)**(m+1) * np.sqrt((n+1+0.5)*n_m/(nm)) * uPmM1_n[n-m+1,z] # P(m+1,n) normalized

        # P(m-1,n-1)
        Pmm1n1,dPmm1n1_dz = sp.special.lpmn(m-1, n-1, x[z])
        uPmm1_n1[n-m,z]     = Pmm1n1[m-1,n-1]
        n_m                 = np.math.factorial(n-1-(m-1))
        nm                  = np.math.factorial(n-1+(m-1))
        Pmm1_n1[n-m,z]     = (-1)**(m-1) * np.sqrt((n-1+0.5)*n_m/(nm)) * uPmm1_n1[n-m,z] # P(m-1,n-1) normalized

        # P(m+1,n-1)
        PmM1n1,dPmM1n1_dz = sp.special.lpmn(m+1, n+1, x[z])
        uPmM1_n1[n-m+2,z]     = PmM1n1[m+1,n+1]
        n_m                 = np.math.factorial(n+1-(m+1))
        nm                  = np.math.factorial(n+1+(m+1))
        PmM1_n1[n-m+2,z]     = (-1)**(m+1) * np.sqrt((n+1+0.5)*n_m/(nm)) * uPmM1_n1[n-m+2,z] # P(m+1,n-1) normalized


    # Computing the remaining Associated Legendre Functions by recurrence relation
    for n in range(m+2,2*N+1):
      Pm_n[n-m,:]    = np.sqrt((4*(n**2)-1)/(n**2-m**2))             * (x*Pm_n[n-m-1,:]-np.sqrt(((n-1)**2 - m**2)/(4*(n-1)**2-1))*Pm_n[n-m-2,:])
      Pmm1_n[n-m,:]  = np.sqrt((4*(n**2)-1)/(n**2-(m-1)**2))         * (x*Pmm1_n[n-m-1,:]-np.sqrt(((n-1)**2-(m-1)**2)/(4*(n-1)**2-1))*Pmm1_n[n-m-2,:])
      Pmm1_n1[n-m,:] = np.sqrt((4*((n-1)**2)-1)/((n-1)**2-(m-1)**2)) * (x*Pmm1_n1[n-m-1,:]-np.sqrt((((n-1)-1)**2-(m-1)**2)/(4*((n-1)-1)**2-1))*Pmm1_n1[n-m-2,:])

    for n in range(m+2,2*N):
      PmM1_n[n-m+1,:] = np.sqrt((4*((n+1)**2)-1)/((n+1)**2-(m+1)**2)) * (x*PmM1_n[n-m,:]-np.sqrt((((n+1)-1)**2-(m+1)**2)/(4*((n+1)-1)**2 -1))*PmM1_n[n-m-1,:])

    for n in range(m+2,2*N-1):
      PmM1_n1[n-m+2,:] = np.sqrt((4*((n+1)**2)-1)/((n+1)**2-(m+1)**2)) * (x*PmM1_n1[n-m+1,:]-np.sqrt((((n+1)-1)**2-(m+1)**2)/(4*((n+1)-1)**2 -1))*PmM1_n1[n-m,:])


    # Derivative of associated Legendre functions with respect to latitude (eq. (3.3))
    dPm_n_dLat = np.zeros((2*N,x.size),dtype=dType)
    for n in range(m,2*N+m):
      dPm_n_dLat[n-m,:] = (1.0/2) * (np.sqrt((n-m)*(n+m+1))*PmM1_n[n-m,:]-np.sqrt((n+m)*(n-m+1))*Pmm1_n[n-m,:]);

    # The term given by eq. (3.4)
    mPm_n_cosLat = np.zeros((2*N,x.size),dtype=dType)
    for n in range(m,2*N+m):
      mPm_n_cosLat[n-m,:] = (1.0/2) * np.sqrt((2*n+1)/(2*n-1)) * (np.sqrt((n+m)*(n+m-1))*Pmm1_n1[n-m,:]+np.sqrt((n-m)*(n-m-1))*PmM1_n1[n-m,:]);


    # The spherical vector harmonics -----------------------------------
    if m==1: print('  - spherical vector harmonics')
    # The spherical vector harmonics will be computed using eqs. (3.1) without the factor e^(i m lambda), since
    # this factor will be canceled in the summation (3.22).
    y1m_n = np.zeros((3,2*N,x.size),dtype=cType)
    y2m_n = np.zeros((3,2*N,x.size),dtype=cType)
    y3m_n = np.zeros((3,2*N,x.size),dtype=cType)

    y3m_n[2] = Pm_n + 0*I
    for n in range(m,2*N+m):
      y1m_n[0,n-m] = I * mPm_n_cosLat[n-m,:] / np.sqrt(n*(n+1))
      y1m_n[1,n-m] = dPm_n_dLat[n-m,:]       / np.sqrt(n*(n+1))
      y2m_n[1,n-m] = I * mPm_n_cosLat[n-m,:] / np.sqrt(n*(n+1))
      y2m_n[0,n-m] = -dPm_n_dLat[n-m,:]      / np.sqrt(n*(n+1))


    # HOUGH vector functions -----------------------------------------
    # The HOUGH vector functions are computed using eq. (3.22) in Swarztrauber and Kasahara (1985)

    if m==1: print('  - HVF: gravity')
    for l in range(1,nLG//2+1):
      for neh in range(1,NEH+1):
        for n in range(1,N+1):
          # ======================== SYMMETRIC SUBSISTEMS ======================
          # ------------------------- Westward gravity -------------------------
          AUX_UVZ[0,m-1,2*l-1-1,neh-1,:]   = I*ABC_WG_sy[m-1,3*n-1-1,l-1,neh-1] * y1m_n[0,2*n-1-1,:] + ABC_WG_sy[m-1,3*n-1,l-1,neh-1] * y2m_n[0,2*n-1,:]
          HOUGH_UVZ[0,m-1,2*l-1-1,neh-1,:] = HOUGH_UVZ[0,m-1,2*l-1-1,neh-1,:] + AUX_UVZ[0,m-1,2*l-1-1,neh-1,:]
          AUX_UVZ[1,m-1,2*l-1-1,neh-1,:]   = I*ABC_WG_sy[m-1,3*n-1-1,l-1,neh-1] * y1m_n[1,2*n-1-1,:] + ABC_WG_sy[m-1,3*n-1,l-1,neh-1] * y2m_n[1,2*n-1,:]
          HOUGH_UVZ[1,m-1,2*l-1-1,neh-1,:] = HOUGH_UVZ[1,m-1,2*l-1-1,neh-1,:] + AUX_UVZ[1,m-1,2*l-1-1,neh-1,:]
          AUX_UVZ[2,m-1,2*l-1-1,neh-1,:]   = -ABC_WG_sy[m-1,3*n-1-2,l-1,neh-1] * y3m_n[2,2*n-1-1,:]
          HOUGH_UVZ[2,m-1,2*l-1-1,neh-1,:] = HOUGH_UVZ[2,m-1,2*l-1-1,neh-1,:] + AUX_UVZ[2,m-1,2*l-1-1,neh-1,:]
          # ------------------------- Eastward gravity -------------------------
          AUX_UVZ[0,m-1,2*l-1-1+nLG,neh-1,:]   = I*ABC_EG_sy[m-1,3*n-1-1,l-1,neh-1] * y1m_n[0,2*n-1-1,:] + ABC_EG_sy[m-1,3*n-1,l-1,neh-1] * y2m_n[0,2*n-1,:]
          HOUGH_UVZ[0,m-1,2*l-1-1+nLG,neh-1,:] = HOUGH_UVZ[0,m-1,2*l-1-1+nLG,neh-1,:] + AUX_UVZ[0,m-1,2*l-1-1+nLG,neh-1,:]
          AUX_UVZ[1,m-1,2*l-1-1+nLG,neh-1,:]   = I*ABC_EG_sy[m-1,3*n-1-1,l-1,neh-1] * y1m_n[1,2*n-1-1,:] + ABC_EG_sy[m-1,3*n-1,l-1,neh-1] * y2m_n[1,2*n-1,:]
          HOUGH_UVZ[1,m-1,2*l-1-1+nLG,neh-1,:] = HOUGH_UVZ[1,m-1,2*l-1-1+nLG,neh-1,:] + AUX_UVZ[1,m-1,2*l-1-1+nLG,neh-1,:]
          AUX_UVZ[2,m-1,2*l-1-1+nLG,neh-1,:]   = -ABC_EG_sy[m-1,3*n-1-2,l-1,neh-1] * y3m_n[2,2*n-1-1,:]
          HOUGH_UVZ[2,m-1,2*l-1-1+nLG,neh-1,:] = HOUGH_UVZ[2,m-1,2*l-1-1+nLG,neh-1,:] + AUX_UVZ[2,m-1,2*l-1-1+nLG,neh-1,:]
          # ======================== ANTISYMMETRIC SUBSISTEMS ==================
          # ------------------------- Westward gravity -------------------------
          AUX_UVZ[0,m-1,2*l-1,neh-1,:]   = I*ABC_WG_asy[m-1,3*n-1,l-1,neh-1] * y1m_n[0,2*n-1,:] + ABC_WG_asy[m-1,3*n-1-2,l-1,neh-1] * y2m_n[0,2*n-1-1,:]
          HOUGH_UVZ[0,m-1,2*l-1,neh-1,:] = HOUGH_UVZ[0,m-1,2*l-1,neh-1,:] + AUX_UVZ[0,m-1,2*l-1,neh-1,:]
          AUX_UVZ[1,m-1,2*l-1,neh-1,:]   = I*ABC_WG_asy[m-1,3*n-1,l-1,neh-1] * y1m_n[1,2*n-1,:] + ABC_WG_asy[m-1,3*n-1-2,l-1,neh-1] * y2m_n[1,2*n-1-1,:]
          HOUGH_UVZ[1,m-1,2*l-1,neh-1,:] = HOUGH_UVZ[1,m-1,2*l-1,neh-1,:] + AUX_UVZ[1,m-1,2*l-1,neh-1,:]
          AUX_UVZ[2,m-1,2*l-1,neh-1,:]   = -ABC_WG_asy[m-1,3*n-1-1,l-1,neh-1] * y3m_n[2,2*n-1,:]
          HOUGH_UVZ[2,m-1,2*l-1,neh-1,:] = HOUGH_UVZ[2,m-1,2*l-1,neh-1,:] + AUX_UVZ[2,m-1,2*l-1,neh-1,:]
          # ------------------------- Eastward gravity -------------------------
          AUX_UVZ[0,m-1,2*l-1+nLG,neh-1,:]   = I*ABC_EG_asy[m-1,3*n-1,l-1,neh-1] * y1m_n[0,2*n-1,:] + ABC_EG_asy[m-1,3*n-1-2,l-1,neh-1] * y2m_n[0,2*n-1-1,:]
          HOUGH_UVZ[0,m-1,2*l-1+nLG,neh-1,:] = HOUGH_UVZ[0,m-1,2*l-1+nLG,neh-1,:] + AUX_UVZ[0,m-1,2*l-1+nLG,neh-1,:]
          AUX_UVZ[1,m-1,2*l-1+nLG,neh-1,:]   = I*ABC_EG_asy[m-1,3*n-1,l-1,neh-1] * y1m_n[1,2*n-1,:] + ABC_EG_asy[m-1,3*n-1-2,l-1,neh-1] * y2m_n[1,2*n-1-1,:]
          HOUGH_UVZ[1,m-1,2*l-1+nLG,neh-1,:] = HOUGH_UVZ[1,m-1,2*l-1+nLG,neh-1,:] + AUX_UVZ[1,m-1,2*l-1+nLG,neh-1,:]
          AUX_UVZ[2,m-1,2*l-1+nLG,neh-1,:]   = -ABC_EG_asy[m-1,3*n-1-1,l-1,neh-1] * y3m_n[2,2*n-1,:]
          HOUGH_UVZ[2,m-1,2*l-1+nLG,neh-1,:] = HOUGH_UVZ[2,m-1,2*l-1+nLG,neh-1,:] + AUX_UVZ[2,m-1,2*l-1+nLG,neh-1,:]


    if m==1: print('  - HVF: rossby')
    for l in range(1,nLR//2+1):
      for neh in range(1,NEH+1):
        for n in range(1,N+1):
          # ======================== SYMMETRIC SUBSISTEMS ======================
          # ------------------------- Westward rossby --------------------------
          AUX_UVZ[0,m-1,2*l-1+2*nLG,neh-1,:]   = I*ABC_WR_sy[m-1,3*n-1-1,l-1,neh-1] * y1m_n[0,2*n-1-1,:] + ABC_WR_sy[m-1,3*n-1,l-1,neh-1] * y2m_n[0,2*n-1,:]
          HOUGH_UVZ[0,m-1,2*l-1+2*nLG,neh-1,:] = HOUGH_UVZ[0,m-1,2*l-1+2*nLG,neh-1,:] + AUX_UVZ[0,m-1,2*l-1+2*nLG,neh-1,:]
          AUX_UVZ[1,m-1,2*l-1+2*nLG,neh-1,:]   = I*ABC_WR_sy[m-1,3*n-1-1,l-1,neh-1] * y1m_n[1,2*n-1-1,:] + ABC_WR_sy[m-1,3*n-1,l-1,neh-1] * y2m_n[1,2*n-1,:]
          HOUGH_UVZ[1,m-1,2*l-1+2*nLG,neh-1,:] = HOUGH_UVZ[1,m-1,2*l-1+2*nLG,neh-1,:] + AUX_UVZ[1,m-1,2*l-1+2*nLG,neh-1,:]
          AUX_UVZ[2,m-1,2*l-1+2*nLG,neh-1,:]   = -ABC_WR_sy[m-1,3*n-1-2,l-1,neh-1] * y3m_n[2,2*n-1-1,:]
          HOUGH_UVZ[2,m-1,2*l-1+2*nLG,neh-1,:] = HOUGH_UVZ[2,m-1,2*l-1+2*nLG,neh-1,:] + AUX_UVZ[2,m-1,2*l-1+2*nLG,neh-1,:]
          # ======================== ANTISYMMETRIC SUBSISTEMS ==================
          # ------------------------- Westward rossby --------------------------
          AUX_UVZ[0,m-1,2*l-1-1+2*nLG,neh-1,:]   = I*ABC_WR_asy[m-1,3*n-1,l-1,neh-1] * y1m_n[0,2*n-1,:] + ABC_WR_asy[m-1,3*n-1-2,l-1,neh-1] * y2m_n[0,2*n-1-1,:]
          HOUGH_UVZ[0,m-1,2*l-1-1+2*nLG,neh-1,:] = HOUGH_UVZ[0,m-1,2*l-1-1+2*nLG,neh-1,:] + AUX_UVZ[0,m-1,2*l-1-1+2*nLG,neh-1,:]
          AUX_UVZ[1,m-1,2*l-1-1+2*nLG,neh-1,:]   = I*ABC_WR_asy[m-1,3*n-1,l-1,neh-1] * y1m_n[1,2*n-1,:] + ABC_WR_asy[m-1,3*n-1-2,l-1,neh-1] * y2m_n[1,2*n-1-1,:]
          HOUGH_UVZ[1,m-1,2*l-1-1+2*nLG,neh-1,:] = HOUGH_UVZ[1,m-1,2*l-1-1+2*nLG,neh-1,:] + AUX_UVZ[1,m-1,2*l-1-1+2*nLG,neh-1,:]
          AUX_UVZ[2,m-1,2*l-1-1+2*nLG,neh-1,:]   = -ABC_WR_asy[m-1,3*n-1-1,l-1,neh-1] * y3m_n[2,2*n-1,:]
          HOUGH_UVZ[2,m-1,2*l-1-1+2*nLG,neh-1,:] = HOUGH_UVZ[2,m-1,2*l-1-1+2*nLG,neh-1,:] + AUX_UVZ[2,m-1,2*l-1-1+2*nLG,neh-1,:]

  # End the zonal wave numbers
  print('End of part II (zonal wave numbers m>0)')

  S_C=S_C.copy()
  Sa_F=Sa_F.copy()

  #  The first symmetric (lowest order) eastward gravity mode is eigenvector of matrix E
  EAST_G_0_sy[0]  = Sa_E[0]
  WEST_R_0_asy[0] = S_C[0]
  for la in range(nLG//2-1):
    EAST_G_0_sy[la+1]  = S_C[la+1]

  for la in range(nLR//2-1):
    WEST_R_0_asy[la+1] = Sa_F[la]

  for la in range(nLR//2):
    WEST_R_0_sy[la]    = Sa_E[la+1]


  EAST_G_0_sy  = EAST_G_0_sy[:nLG//2]
  WEST_R_0_asy = WEST_R_0_asy[:nLR//2]


  # end of calculations

  out=dict(HOUGH_UVZ=HOUGH_UVZ,HOUGH_0_UVZ=HOUGH_0_UVZ,
           # westward - eddies:
           WEST_G_sy=WEST_G_sy,WEST_G_asy=WEST_G_asy,WEST_R_sy=WEST_R_sy,WEST_R_asy=WEST_R_asy,
           # eastward - eddies:
           EAST_G_sy=EAST_G_sy,EAST_G_asy=EAST_G_asy,
           # westward - zonal mean:
           WEST_G_0_sy=WEST_G_0_sy,WEST_G_0_asy=WEST_G_0_asy,WEST_R_0_sy=WEST_R_0_sy,WEST_R_0_asy=WEST_R_0_asy,
           # eastward - zonal mean:
           EAST_G_0_sy=EAST_G_0_sy, EAST_G_0_asy=EAST_G_0_asy)


  return out, truncation_order, x


def calc_CD(p_n,r_n,add=0):
  '''
  add=0  --> C
  add=1  --> D

  C, Eq. (4.8) in Swarztrauber and Kasahara (1985)
  D, Eq. (4.11) in Swarztrauber and Kasahara (1985)
  '''
  nn,NEH=r_n.shape
  N=nn//2

  C_dd = np.zeros((N-1,NEH),dtype=dType)
  C_md = np.zeros((N,NEH)  ,dtype=dType)

  # Upper (and lower) diagonal elements:
  for k in range(N-1):
    C_dd[k,:] = p_n[2*k+add] * p_n[2*k+1+add]

  if add==0:
    # First index for r_n is zero while for p_n is one.
    C_md[0] = r_n[0] * r_n[0] + p_n[0] * p_n[0]
    i0=1
  else: i0=0

  # Main diagonal elements
  for k in range(i0,N):
    C_md[k] = r_n[2*k+add] * r_n[2*k+add] + p_n[2*k-1+add] * p_n[2*k-1+add] + p_n[2*k+add] * p_n[2*k+add]

  # Creating the tridiagonal matrix and calculating it's eigenvalues/eigenvectors.
  C      = np.zeros((N,N,NEH),dtype=dType)
  S_C    = np.zeros((N,NEH)  ,dtype=dType)
  S_Caux = np.zeros((N,NEH)  ,dtype=dType)
  U_C    = np.zeros((N,N,NEH),dtype=dType)
  IC2sort = np.zeros((N,NEH) ,dtype=int)

  for gi in range(0,NEH):
    # Tridiagonal matrix
    C[:,:,gi] = np.diag(C_dd[:,gi],k=-1) + np.diag(C_md[:,gi]) + np.diag(C_dd[:,gi],k=1)
    # Eigenvalues/eigenvectors of C/D [Eq. (4.9/4.12) in Swarztrauber and Kasahara (1985)]
    auxAUXc,auxU_C = np.linalg.eigh(np.squeeze(C[:,:,gi]))
    U_C[:,:,gi]    = auxU_C
    S_Caux[:,gi]   = auxAUXc
    IC2sort[:,gi]  = np.argsort(S_Caux[:,gi],axis=0) # indices that would sort the eigenvalues
    S_Caux[:,gi]   = S_Caux[IC2sort[:,gi],gi]        # Sorting the eigenvalues
    U_C[:,:,gi]    = U_C[:,IC2sort[:,gi],gi]         # Sorting the corresponding eigenvectors

  S_C = np.sqrt(S_Caux);   # Frequencies (Sigma)

  return C,S_C,U_C
