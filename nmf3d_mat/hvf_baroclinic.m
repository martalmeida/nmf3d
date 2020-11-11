function [out,truncation_order,x]=hvf_baroclinic(hk,M,nLR,nLG,latType,dlat)
%  Compute the Hough vector functions as described in the paper of Swarztrauber and Kasahara (1985).
%  Baroclinic mode. hk(1) can't be inf.
%
%  Part I: The frequencies and the Hough functions are computed for zonal wave number m = 0.
%  Part II: The frequencies and the Hough functions are computed for zonal wave numbers m > 0.
%
%  See hough_functions

constants;

disp('- HVF baroclinic -');

L = nLR + 2*nLG; % Total number of meridional modes used in the expansion (should be even)

% The equivalent heights
NEH=length(hk);

% Dimensionless constant (gamma) computed from (2.8) of Swarztrauber and Kasahara (1985):
% gamma=sqrt(g*hk)/(2*Er*Om), where hk are the equivalent heights obtained as the
% solution of the Vertical Structure Equations
Ga = sqrt(const.g*hk) / (2*const.Er*const.Om);

% Lamb's parameter (epson)
Ep = Ga.^-2;

% Truncation order for the expansion in terms of the spherical vector harmonics
N = max([20,L,ceil(max(sqrt(Ep)))]); % why 20? see Swarztrauber and A. Kasahara (1985), pg 481.
truncation_order=N; % returned because is needed for the barotropic if ws0 is True

% Total number of eigenvalues/eigenvectors for matrices A and B.
maxN = 3*N;

% Latitude points
if isequal(latType,'linear')
  LAT = -90.:dlat:90.;
  %x   = sin(LAT*pi/180.);
  x   = sind(LAT);
elseif isequal(latType,'gaussian')
  [x,w] = lgwt(dlat,-1,1);
end

disp('Part I');
% PART I ----------------------------------------------------------
% For zonal wave number m=0, the frequencies and associated horizontal
% stucture functions are computed as the eigenvalues/eigenvectors of
% matrices C, D, E and F given in Swarztrauber and Kasahara (1985).
% Since the frequencies are not determined in triplets, matrices
% C, D, E and F have dimensions N x N.

% Matrices C and D =================================================

% p_n, Eq. (4.2) in Swarztrauber and Kasahara (1985)
p_n = zeros(2*N);
for k=1:2*N
  p_n(k) = sqrt((k-1.)*(k+1.)/((2*k-1.)*(2*k+1.)));
end

% r_n, Eq. (3.26) in Swarztrauber and Kasahara (1985)
r_n = zeros(2*N,NEH);
for gi=1:NEH
  for k=0:2*N-1
    r_n(k+1,gi) = Ga(gi)*sqrt(k*(k+1));
  end
end

disp('  - Matrix C');
[C,S_C,U_C] = calc_CD(p_n,r_n,0);

disp('  - Matrix D');
[D,S_D,U_D] = calc_CD(p_n,r_n,1);

% Matrices E and F =================================================

% Term  n(n+1) of eq. (4.19) in Swarztrauber and Kasahara (1985)
n_n1 = zeros(2*N+1,1);
for n=1:2*N+1
  n_n1(n) = n*(n+1);
end

% Terms d_n and e_n of eqs. (4.18) in Swarztrauber and Kasahara (1985)
d_n = zeros(2*N+2,NEH);
e_n = zeros(2*N+2,NEH);
for gi=1:NEH
  d_n(:,gi) = ([0:2*N+1]-1)./(Ga(gi)*sqrt((2*[0:2*N+1]-1).*(2*[0:2*N+1]+1)));
  e_n(:,gi) = ([0:2*N+1]+2)./(Ga(gi)*sqrt((2*[0:2*N+1]+1).*(2*[0:2*N+1]+3)));
end

% Matrix E is given by eq. (4.21) in Swarztrauber and Kasahara (1985)
disp('  - Matrix E')

E_dd = zeros(N,NEH);   % complex
E_md = zeros(N+1,NEH); % real

% Upper (and lower) diagonal elements
for k=1:N
   E_dd(k,:) = d_n(2*k-1,:).*e_n(2*k-1,:);
end

E_md(1,:) = d_n(1,:).*d_n(1,:);
% Main diagonal elements for Matrix E.
for k=2:N+1
   E_md(k,:) =  n_n1(2*k-3) + e_n(2*k-3,:).*e_n(2*k-3,:) + d_n(2*k-1,:).*d_n(2*k-1,:);
end

% Creating the tridiagonal matrix E and calculating it's eigenvalues/eigenvectors
% Creating matrix diag(i,1,...,1) for diagonal similarity transform of matrix E
aux_e = eye(N+1);
aux_e(1,1)=sqrt(-1);
U_E     = zeros(N+1,N+1,NEH); % complex
S_Eaux  = zeros(N+1,NEH);

for gi=1:NEH
  % Tridiagonal matrix E
  E = diag(E_dd(:,gi),-1) + diag(E_md(:,gi),0) + diag(E_dd(:,gi),1);

  % The eigenvectors/eigenvalues of E [Eq. (4.22) in Swarztrauber and Kasahara (1985)]
  [eig_vec,eig_val]=eig(aux_e'*E*aux_e);
  [eig_val,isort]=sort(diag(eig_val));
  eig_vec=eig_vec(:,isort);

  U_E(:,:,gi)  = aux_e*eig_vec; % Multliplies sorted eigenvectors by diag(i,1,...,1)
  S_Eaux(:,gi) = eig_val;
end

% Frequencies Sigma_a, Eqs. (4.18) in Swarztrauber and Kasahara (1985)
Sa_E = -1./S_Eaux;


% Matrix F is given by eq. (4.24) in Swarztrauber and Kasahara (1985)
disp('  - Matrix F');

F_dd = zeros(N-1,NEH);
F_md = zeros(N,NEH);

% Upper (and lower) diagonal elements for Matrix F
for k=2:N
  F_dd(k-1,:) = d_n(2*k,:).*e_n(2*k,:);
end

% Main diagonal elements for Matrix F
for k=1:N
  F_md(k,:) =  n_n1(2*k) + e_n(2*k,:).*e_n(2*k,:) + d_n(2*k+2,:).*d_n(2*k+2,:);
end

% Matrix F, eigenvalues/eigenvectors
U_F    = zeros(N,N,NEH);
S_Faux = zeros(N,NEH);

for gi=1:NEH
  % Tridiagonal matrix F
  F = diag(F_dd(:,gi),-1) + diag(F_md(:,gi),0) + diag(F_dd(:,gi),1);

  % The eigenvectors/eigenvalues of F [Eq. (4.25) in Swarztrauber and Kasahara (1985)]
  [eig_vec,eig_val]=eig(F);
  [eig_val,isort]=sort(diag(eig_val));
  eig_vec=eig_vec(:,isort);

  U_F(:,:,gi)  = eig_vec;
  S_Faux(:,gi) = eig_val;
end

% Frequencies Sigma_a, Eqs. (4.18) in Swarztrauber and Kasahara (1985)
Sa_F = -1./S_Faux;

% Selecting the frequencies and the coefficients for the vector harmonic expansion
% The eigenvalues are classified as:
% Westward gravity    => from eigenvalues of matrices C and D
% Westward Rotational => from eigenvalues of matrices E and F
% Eastward gravity    => from eigenvalues of matrices C and D

% Coefficients An, Bn and Cn =======================================
disp('  - Coeffs An, Bn, Cn');

% Case 1 : Gravity Modes
%
% Coefficients An are obtained from eigenvectors of matrices C and D, and
% coefficients Bn and Cn are obtained from An using eqs. (4.4) and (4.5)

% Arrays for the frequencies (eigenvalues)
%WEST_G_0_sy  = zeros(nLG/2,NEH);
%WEST_G_0_asy = zeros(nLG/2,NEH);
%EAST_G_0_sy  = zeros(nLG/2,NEH);
%EAST_G_0_asy = zeros(nLG/2,NEH);

% Arrays for the coefficients (eigenvectors)
% The coefficients are stored in columns as:
% symmetric subsystem     --> (C0, A0, B1, C2, A2, B3, ...)
% antisymmetric subsystem --> (B0, C1, A1, B2, C3, A3, ...)
ABC_WG_0_sy  = zeros(maxN,nLG/2,NEH);
ABC_WG_0_asy = zeros(maxN,nLG/2,NEH);
ABC_EG_0_sy  = zeros(maxN,nLG/2,NEH);
ABC_EG_0_asy = zeros(maxN,nLG/2,NEH);

% An_G_0_sy    = zeros(N,nLG/2,NEH);
% An_G_0_asy   = zeros(N,nLG/2,NEH);
Bn_G_0_sy    = zeros(N,nLG/2,NEH);
Bn_G_0_asy   = zeros(N,nLG/2,NEH);
Cn_G_0_sy    = zeros(N,nLG/2,NEH);
Cn_G_0_asy   = zeros(N,nLG/2,NEH);

% Frequencies (eigenvalues)
WEST_G_0_sy  = -S_C(1:nLG/2,:);
WEST_G_0_asy = -S_D(1:nLG/2,:);

EAST_G_0_sy  =  S_C(1:nLG/2,:);
EAST_G_0_asy =  S_D(1:nLG/2,:);


% Coefficients An (eigenvectors)
An_G_0_sy  = U_C(:,1:nLG/2,:);  % symmetric subsystem -> [A0, A2, A4, ... , A2N-2] <- from matrix C
An_G_0_asy = U_D(:,1:nLG/2,:);  % antisymmetric subsystem -> [A1, A3, A5, ... , A2N-1] <- from matrix D


% Computation of Bn and Cn
% Cn => Eq. (4.5) in Swarztrauber and Kasahara (1985)
r_n_aux = zeros(2*N,nLG/2,NEH);
for s=1:nLG/2
  r_n_aux(:,s,:) = r_n;
end

for n=1:N
  for l=1:nLG/2
    % symmetric subsystem -> (C0, C2, C4, ... , C2N-2) <- from matrix C
    if l>1 % to avoid division by zero warning (i.e. EAST_G_0_sy[l=0,:]=0)
      Cn_G_0_sy(n,l,:) = squeeze(r_n_aux(2*n-1,l,:))' .* squeeze(An_G_0_sy(n,l,:))' ./ EAST_G_0_sy(l,:);
    end

    % antisymmetric subsystem -> (C1, C3, C5, ... , C2N-1) <- from matrix D
    Cn_G_0_asy(n,l,:) = squeeze(r_n_aux(2*n,l,:))' .* squeeze(An_G_0_asy(n,l,:))' ./ EAST_G_0_asy(l,:);
 end
end

% Bn => Eq. (4.4) in Swarztrauber and Kasahara (1985)
% Last Bn for the symmetric subsystem must be computed from eq. (4.3)
for n=1:N-1
  for l=2:nLG/2 % starts at 2 to avoid division by zero warning, see above
    % symmetric subsystem -> (B1, B3, B5, ... ) <- from matrix C
    Bn_G_0_sy(n,l,:) = (p_n(2*n-1).*squeeze(An_G_0_sy(n,l,:))'+p_n(2*n).*squeeze(An_G_0_sy(n+1,l,:))')./EAST_G_0_sy(l,:);
  end
end

% Last Bn for the symmetric subsystem computed from eq 4.3
for l=1:nLG/2
   Bn_G_0_sy(N,l,:) = (EAST_G_0_sy(l,:).*squeeze(An_G_0_sy(N,l,:))'-r_n(2*N-1,:).*squeeze(Cn_G_0_sy(N,l,:))'-p_n(2*N-2).* ...
                       squeeze(Bn_G_0_sy(N-1,l,:))')./p_n(2*N-1);
end

Bn_G_0_asy(1,:,:) = 0; % B0 = 0 from eq 4.4
for n=2:N
  for l=1:nLG/2
    % antisymmetric subsystem -> (B0, B2, B4, ... ) <- from matrix D
    Bn_G_0_asy(n,l,:) = (p_n(2*n-2).*squeeze(An_G_0_asy(n-1,l,:))'+p_n(2*n-1).*squeeze(An_G_0_asy(n,l,:))')./EAST_G_0_asy(l,:);
  end
end

% Storing the coefficients in columns as:
% symmetric subsystem     --> (C0, A0, B1, C2, A2, B3, ...)
% antisymmetric subsystem --> (B0, C1, A1, B2, C3, A3, ...)

n1=1;
for n=1:3:maxN
  % Eastward gravity:
  % - symmetric subsystem:
  ABC_EG_0_sy(n,:,:)    = Cn_G_0_sy(n1,:,:);
  ABC_EG_0_sy(n+1,:,:)  = An_G_0_sy(n1,:,:);
  ABC_EG_0_sy(n+2,:,:)  = Bn_G_0_sy(n1,:,:);
  % - antisymmetric subsystem:
  ABC_EG_0_asy(n,:,:)   = Bn_G_0_asy(n1,:,:);
  ABC_EG_0_asy(n+1,:,:) = Cn_G_0_asy(n1,:,:);
  ABC_EG_0_asy(n+2,:,:) = An_G_0_asy(n1,:,:);
  %
  % Westward gravity:
  % - symmetric subsystem:
  ABC_WG_0_sy(n,:,:)    = Cn_G_0_sy(n1,:,:);
  ABC_WG_0_sy(n+1,:,:)  = -An_G_0_sy(n1,:,:);
  ABC_WG_0_sy(n+2,:,:)  = Bn_G_0_sy(n1,:,:);
  % - antisymmetric subsystem:
  ABC_WG_0_asy(n,:,:)   = Bn_G_0_asy(n1,:,:);
  ABC_WG_0_asy(n+1,:,:) = Cn_G_0_asy(n1,:,:);
  ABC_WG_0_asy(n+2,:,:) = -An_G_0_asy(n1,:,:);
  %
  n1=n1+1;
end


% There are two modes with zero frequency which are classified as gravity modes.
% The first is obtained by noting that U=(1,0,0,...)^T is an eigenvector of C,
% which corresponds to eigenvalue sigma^2=0. Therefore the first westward gravity
% mode is identified with A0=1 and all other coeficients are zero.
ABC_WG_0_sy(:,1,:) = 0; % all other coeficients are zero.
ABC_WG_0_sy(2,1,:) = 1; % A0=1 (cf. p.475 of Swarztrauber and Kasahara (1985)).


% Case 2 : Rossby(rotational) Modes
%
% Coefficients An are all zero
% Coefficients Bn_til are obtained from eigenvectors of matrices E and F
% Bn and Cn are obtained from Bn_til using eqs. (4.18) and (4.26)
% (4.18) -> Bn=sqrt(n(n+1))*Bn_til
% (4.26) -> Cn=-dn*Bn-1_til - en*Bn+1_til.

% Frequencies (eigenvalues)
WEST_R_0_sy  = Sa_E(1:nLR/2,:);
WEST_R_0_asy = Sa_F(1:nLR/2,:);


% Arrays for the coefficients (eigenvectors)
% The coefficients are stored in columns as:
% symmetric subsystem     --> (C0, A0, B1, C2, A2, B3, ...)
% antisymmetric subsystem --> (B0, C1, A1, B2, C3, A3, ...)
ABC_WR_0_sy  = zeros(maxN,nLR/2+1,NEH);
An_R_0_sy    = zeros(N+1,nLR/2+1,NEH);
auxBn_R_0_sy = zeros(N+1,nLR/2+1,NEH);
Bn_R_0_sy    = zeros(N+1,nLR/2+1,NEH);
Cn_R_0_sy    = zeros(N+1,nLR/2+1,NEH);
%Bn_til_0_sy  = zeros(N+1,nLR/2+1,NEH); % complex
Bn_til_0_sy  = U_E(:,1:nLR/2+1,:);  % symmetric subsystem --> [Bn_til[-1], Bn_til[1], Bn_til[3], ... ] <-- matrix E

ABC_WR_0_asy  = zeros(maxN,nLR/2,NEH);
An_R_0_asy    = zeros(N,nLR/2,NEH);
auxBn_R_0_asy = zeros(N,nLR/2,NEH);
Bn_R_0_asy    = zeros(N,nLR/2,NEH);
Cn_R_0_asy    = zeros(N,nLR/2,NEH);
%Bn_til_0_asy  = zeros(N,nLR/2,NEH);
Bn_til_0_asy = U_F(:,1:nLR/2,:); % antisymmetric subsystem --> [ Bn_til[2], Bn_til[4], Bn_til[6], ... ] <-- matrix F


% Computation of Bn and Cn
aux_sy = zeros(N+1,1);
nn1=1;
for n=-1:2:2*N
  aux_sy(nn1) = sqrt(n*(n+1));
  nn1=nn1+1;
end

aux_asy = zeros(N,1);
nn2=1;
for n=2:2:2*N
  aux_asy(nn2) = sqrt(n*(n+1));
  nn2=nn2+1;
end

% symmetric subsystem --> (B-1, B1, B3, B5 ... ) <-- from matrix E (with N+1 elements)
for n=1:N+1 % Element N+1 of Bn_til_0_sy is used to compute last element of Bn_R_0_sy
  % Eq. (4.18) in Swarztrauber and Kasahara (1985)
  auxBn_R_0_sy(n,:,:) = aux_sy(n)*Bn_til_0_sy(n,:,:);
end

% The Bn's to be stored are (B1, B3, B5 ... ), hence:
Bn_R_0_sy(1:N,:,:) = auxBn_R_0_sy(2:N+1,:,:);

% symmetric subsystem --> (C0, C2, C4, ... ) <-- from matrix E (with N+1 elements)
for n=1:N % Only the first N elements will be used
   for k=1:NEH
      % Eq. (4.26) in Swarztrauber and Kasahara (1985)
      Cn_R_0_sy(n,:,k) = -d_n(2*n-1,k)*Bn_til_0_sy(n,:,k) - e_n(2*n-1,k)*Bn_til_0_sy(n+1,:,k);
   end
end

for n=1:N
   for k=1:NEH
      % antisymmetric subsystem --> (B2, B4, B6, ... ) <-- from matrix F (with N elements)
      auxBn_R_0_asy(n,:,k) = aux_asy(n)*Bn_til_0_asy(n,:,k);
      if n==1
        Cn_R_0_asy(n,:,k)    = -e_n(2*n,k)*Bn_til_0_asy(n,:,k);
      else
        Cn_R_0_asy(n,:,k)    = -d_n(2*n,k)*Bn_til_0_asy(n-1,:,k) - e_n(2*n,k)*Bn_til_0_asy(n,:,k);
      end
   end
end

% The Bn's to be stored are (B0=0, B2, B4 ... ), hence:
Bn_R_0_asy(2:N,:,:) = auxBn_R_0_asy(1:N-1,:,:);
Bn_R_0_asy(1,:,:)   = 0;


% Storing the coefficients in columns as:
% symmetric subsystem     --> (C0, A0, B1, C2, A2, B3, ...)
% antisymmetric subsystem --> (B0, C1, A1, B2, C3, A3, ...)
n1=1;
for n=1:3:maxN
  % _____________Westward rotational_______________
  % ------------ symmetric subsystem ------------ #
  ABC_WR_0_sy(n,:,:)    = Cn_R_0_sy(n1,:,:);
  % An coefficients are all zero
  ABC_WR_0_sy(n+2,:,:)  = Bn_R_0_sy(n1,:,:);

  % ---------- antisymmetric subsystem ---------- #
  ABC_WR_0_asy(n,:,:)   = Bn_R_0_asy(n1,:,:);
  ABC_WR_0_asy(n+1,:,:) = Cn_R_0_asy(n1,:,:);
  % An coefficients are all zero
  % ____________________________________________
  n1=n1+1;
end

% The eastward gravity mode of lowest order
% is eigenvector of matrix E. Therefore:
ABC_EG_0_sy(:,1,:)=ABC_WR_0_sy(:,1,:);

% Normalising the coefficients -------------------------------------
NormaWG_sy  = zeros(nLG/2,NEH);
NormaWG_asy = zeros(nLG/2,NEH);
NormaEG_sy  = zeros(nLG/2,NEH);
NormaEG_asy = zeros(nLG/2,NEH);
NormaWR_sy  = zeros(nLR/2+1,NEH);
NormaWR_asy = zeros(nLR/2,NEH);

for la=1:nLG/2
  for nh=1:NEH
    NormaWG_sy(la,nh)  = norm(squeeze(ABC_WG_0_sy(:,la,nh)));
    NormaWG_asy(la,nh) = norm(squeeze(ABC_WG_0_asy(:,la,nh)));
    NormaEG_sy(la,nh)  = norm(squeeze(ABC_EG_0_sy(:,la,nh)));
    NormaEG_asy(la,nh) = norm(squeeze(ABC_EG_0_asy(:,la,nh)));
  end
end

for la=1:nLR/2
   for nh=1:NEH
      NormaWR_sy(la,nh)  = norm(squeeze(ABC_WR_0_sy(:,la,nh)));
      NormaWR_asy(la,nh) = norm(squeeze(ABC_WR_0_asy(:,la,nh)));
   end
end

for nh=1:NEH
  NormaWR_sy(nLR/2+1,nh)  = norm(squeeze(ABC_WR_0_sy(:,nLR/2+1,nh)));
end

NormaWG_sy=reshape(NormaWG_sy,[1 size(NormaWG_sy)]);
NormaWG_sy=repmat(NormaWG_sy,size(ABC_WG_0_sy,1),1,1);

NormaWG_asy=reshape(NormaWG_asy,[1 size(NormaWG_asy)]);
NormaWG_asy=repmat(NormaWG_asy,size(ABC_WG_0_asy,1),1,1);

NormaEG_sy=reshape(NormaEG_sy,[1 size(NormaEG_sy)]);
NormaEG_sy=repmat(NormaEG_sy,size(ABC_EG_0_sy,1),1,1);

NormaEG_asy=reshape(NormaEG_asy,[1 size(NormaEG_asy)]);
NormaEG_asy=repmat(NormaEG_asy,size(ABC_EG_0_asy,1),1,1);

NormaWR_sy=reshape(NormaWR_sy,[1 size(NormaWR_sy)]);
NormaWR_sy=repmat(NormaWR_sy,size(ABC_WR_0_sy,1),1,1);

NormaWR_asy=reshape(NormaWR_asy,[1 size(NormaWR_asy)]);
NormaWR_asy=repmat(NormaWR_asy,size(ABC_WR_0_asy,1),1,1);

ABC_WG_0_sy  = ABC_WG_0_sy./NormaWG_sy;
ABC_WG_0_asy = ABC_WG_0_asy./NormaWG_asy;
ABC_EG_0_sy  = ABC_EG_0_sy./NormaEG_sy;
ABC_EG_0_asy = ABC_EG_0_asy./NormaEG_asy;
ABC_WR_0_sy  = ABC_WR_0_sy./NormaWR_sy;
ABC_WR_0_asy = ABC_WR_0_asy./NormaWR_asy;

% Because the westward gravity modes have been obtained as minus the square root of the
% eigenvalues of C and D, their order may be reversed (optionally).
%  for n in np.arange(0,maxN):
%     ABC_WG_0_sy[n,:,:]  = fliplr(squeeze(ABC_WG_0_sy[n,:,:]));
%     ABC_WG_0_asy[n,:,:] = fliplr(squeeze(ABC_WG_0_asy[n,:,:]));


% Storing the symmetric and antisymmetric coefficients for the Rossby modes in one array
ABC_WR_0 = zeros(maxN,nLR,NEH);

% Similarly to the first westward gravity mode (which is identified with A0=1 and all
% other coeficients zero), the first westward rotational mode is identified with B0=1
%  and all other coeficients zero.
ABC_WR_0(1,1,:) = 1; % B0=1 and all other coeficients are zero.

for l=2:nLR/2
  ABC_WR_0(:,2*l-2,:) = ABC_WR_0_sy(:,l,:);
  ABC_WR_0(:,2*l-1,:) = ABC_WR_0_asy(:,l-1,:);
end
ABC_WR_0(:,nLR,:) = ABC_WR_0_sy(:,nLR/2+1,:);


% HOUGH VECTOR FUNCTIONS ============================================
%  Normalized Associated Legendre Functions (Pm_n) ------------------
%  The associated Legendre functions are evaluated for degrees n=0,...,2*N and orders M = 0 and 1, for
%  each element of X. N must be a scalar integer and X must contain real values between -1 <= x <= 1.

% M=0
disp('  - Normalized Associated Legendre Functions - M=0');
P0_n  = zeros(2*N,length(x)); % normalized
for n=0:2*N-1
  AUX = legendre(n,x,'norm');
  P0_n(n+1,:) = AUX(1,:); % P(0,n)
end

% M=1
disp('  - Normalized Associated Legendre Functions - M=1');
P1_n  = zeros(2*N,length(x)); % normalized
%P1_n(1,:) = 0;  % P(1,0)=0
for n=1:2*N-1
   AUX1 = legendre(n,x,'norm');
   P1_n(n+1,:) = AUX1(2,:); % P(1,n)
end


% HOUGH vector functions --------
% computed using eq. (3.22) in Swarztrauber and Kasahara (1985)
%
% Arrays for the Hough functions
HOUGH_0_UVZ = zeros(3,L,NEH,length(x)); % (complex) Hough functions for n=0.
AUX_0_UVZ   = zeros(3,L,NEH,length(x)); % (complex) Auxiliar array for summation (3.22)

% GRAVITY MODES
disp('  - HVF: gravity modes');

I=const.I;
for l=1:nLG/2
  for neh=1:NEH
    for n=1:N
      % ========================== SYMMETRIC SUBSISTEMS ======================
      % --------------------------- Westward gravity -------------------------
      % ----------------------------- Component 1 ----------------------------
      AUX_0_UVZ(1,2*l-1,neh,:)   = - ABC_WG_0_sy(3*n,l,neh) .* P1_n(2*n,:);
      HOUGH_0_UVZ(1,2*l-1,neh,:) = HOUGH_0_UVZ(1,2*l-1,neh,:) + AUX_0_UVZ(1,2*l-1,neh,:);
      % ----------------------------- Component 2 ----------------------------
      AUX_0_UVZ(2,2*l-1,neh,:)   = I*ABC_WG_0_sy(3*n-1,l,neh) .* P1_n(2*n-1,:);
      HOUGH_0_UVZ(2,2*l-1,neh,:) = HOUGH_0_UVZ(2,2*l-1,neh,:) + AUX_0_UVZ(2,2*l-1,neh,:);
      % ----------------------------- Component 3 ----------------------------
      AUX_0_UVZ(3,2*l-1,neh,:)   = -ABC_WG_0_sy(3*n-2,l,neh) .* P0_n(2*n-1,:);
      HOUGH_0_UVZ(3,2*l-1,neh,:) = HOUGH_0_UVZ(3,2*l-1,neh,:) + AUX_0_UVZ(3,2*l-1,neh,:);
      % --------------------------- Eastward gravity -------------------------
      % ---------------------------- Component 1 -----------------------------
      AUX_0_UVZ(1,2*l-1+nLG,neh,:)   = - ABC_EG_0_sy(3*n,l,neh) .* P1_n(2*n,:);
      HOUGH_0_UVZ(1,2*l-1+nLG,neh,:) = HOUGH_0_UVZ(1,2*l-1+nLG,neh,:) + AUX_0_UVZ(1,2*l-1+nLG,neh,:);
      % ----------------------------- Component 2 ----------------------------
      AUX_0_UVZ(2,2*l-1+nLG,neh,:)   = I*ABC_EG_0_sy(3*n-1,l,neh) .* P1_n(2*n-1,:);
      HOUGH_0_UVZ(2,2*l-1+nLG,neh,:) = HOUGH_0_UVZ(2,2*l-1+nLG,neh,:) + AUX_0_UVZ(2,2*l-1+nLG,neh,:);
      % ----------------------------- Component 3 ----------------------------
      AUX_0_UVZ(3,2*l-1+nLG,neh,:)   = -ABC_EG_0_sy(3*n-2,l,neh) .* P0_n(2*n-1,:);
      HOUGH_0_UVZ(3,2*l-1+nLG,neh,:) = HOUGH_0_UVZ(3,2*l-1+nLG,neh,:) + AUX_0_UVZ(3,2*l-1+nLG,neh,:);
      % ========================== ANTISYMMETRIC SUBSISTEMS ==================
      % --------------------------- Westward gravity -------------------------
      % ---------------------------- Component 1 -----------------------------
      AUX_0_UVZ(1,2*l,neh,:)   = - ABC_WG_0_asy(3*n-2,l,neh) .* P1_n(2*n-1,:);
      HOUGH_0_UVZ(1,2*l,neh,:) = HOUGH_0_UVZ(1,2*l,neh,:) + AUX_0_UVZ(1,2*l,neh,:);
      % ---------------------------- Component 2 -----------------------------
      AUX_0_UVZ(2,2*l,neh,:)   = I*ABC_WG_0_asy(3*n,l,neh) .* P1_n(2*n,:);
      HOUGH_0_UVZ(2,2*l,neh,:) = HOUGH_0_UVZ(2,2*l,neh,:) + AUX_0_UVZ(2,2*l,neh,:);
      % ---------------------------- Component 3 -----------------------------
      AUX_0_UVZ(3,2*l,neh,:)   = -ABC_WG_0_asy(3*n-1,l,neh) .* P0_n(2*n,:);
      HOUGH_0_UVZ(3,2*l,neh,:) = HOUGH_0_UVZ(3,2*l,neh,:) + AUX_0_UVZ(3,2*l,neh,:);
      % --------------------------- Eastward gravity -------------------------
      % ---------------------------- Component 1 -----------------------------
      AUX_0_UVZ(1,2*l+nLG,neh,:)   = - ABC_EG_0_asy(3*n-2,l,neh) .* P1_n(2*n-1,:);
      HOUGH_0_UVZ(1,2*l+nLG,neh,:) = HOUGH_0_UVZ(1,2*l+nLG,neh,:) + AUX_0_UVZ(1,2*l+nLG,neh,:);
      % ---------------------------- Component 2 -----------------------------
      AUX_0_UVZ(2,2*l+nLG,neh,:)   = I*ABC_EG_0_asy(3*n,l,neh) .* P1_n(2*n,:);
      HOUGH_0_UVZ(2,2*l+nLG,neh,:) = HOUGH_0_UVZ(2,2*l+nLG,neh,:) + AUX_0_UVZ(2,2*l+nLG,neh,:);
      % ---------------------------- Component 3 -----------------------------
      AUX_0_UVZ(3,2*l+nLG,neh,:)   = -ABC_EG_0_asy(3*n-1,l,neh) .* P0_n(2*n,:);
      HOUGH_0_UVZ(3,2*l+nLG,neh,:) = HOUGH_0_UVZ(3,2*l+nLG,neh,:) + AUX_0_UVZ(3,2*l+nLG,neh,:);
    end
  end
end


% ROSSBY MODES
disp('  - HVF: rossby modes');

for l=1:nLR/2
  for neh=1:NEH
    for n=1:N
      % ================================= SYMMETRIC SUBSISTEMS ================================
      % --------------------------------- Westward rossby -------------------------------------
      % ----------------------------------- Component 1 ---------------------------------------
      AUX_0_UVZ(1,2*l+2*nLG,neh,:)   = - ABC_WR_0(3*n,2*l,neh) .* P1_n(2*n,:);
      HOUGH_0_UVZ(1,2*l+2*nLG,neh,:) = HOUGH_0_UVZ(1,2*l+2*nLG,neh,:) + AUX_0_UVZ(1,2*l+2*nLG,neh,:);
      % ----------------------------------- Component 2 ---------------------------------------
      AUX_0_UVZ(2,2*l+2*nLG,neh,:)   = I*ABC_WR_0(3*n-1,2*l,neh) .* P1_n(2*n-1,:);
      HOUGH_0_UVZ(2,2*l+2*nLG,neh,:) = HOUGH_0_UVZ(2,2*l+2*nLG,neh,:) + AUX_0_UVZ(2,2*l+2*nLG,neh,:);
      % ----------------------------------- Component 3 ---------------------------------------
      AUX_0_UVZ(3,2*l+2*nLG,neh,:)   = -ABC_WR_0(3*n-2,2*l,neh) .* P0_n(2*n-1,:);
      HOUGH_0_UVZ(3,2*l+2*nLG,neh,:) = HOUGH_0_UVZ(3,2*l+2*nLG,neh,:) + AUX_0_UVZ(3,2*l+2*nLG,neh,:);
      % ================================= ANTISYMMETRIC SUBSISTEMS ============================
      % --------------------------------- Westward rossby -------------------------------------
      % ----------------------------------- Component 1 ---------------------------------------
      AUX_0_UVZ(1,2*l-1+2*nLG,neh,:)   = - ABC_WR_0(3*n-2,2*l-1,neh) .* P1_n(2*n-1,:);
      HOUGH_0_UVZ(1,2*l-1+2*nLG,neh,:) = HOUGH_0_UVZ(1,2*l-1+2*nLG,neh,:) + AUX_0_UVZ(1,2*l-1+2*nLG,neh,:);
      % ------------------------------------ Component 2 --------------------------------------
      AUX_0_UVZ(2,2*l-1+2*nLG,neh,:)   = I*ABC_WR_0(3*n,2*l-1,neh) .* P1_n(2*n,:);
      HOUGH_0_UVZ(2,2*l-1+2*nLG,neh,:) = HOUGH_0_UVZ(2,2*l-1+2*nLG,neh,:) + AUX_0_UVZ(2,2*l-1+2*nLG,neh,:);
      % ----------------------------------- Component 3 ---------------------------------------
      AUX_0_UVZ(3,2*l-1+2*nLG,neh,:)   = -ABC_WR_0(3*n-1,2*l-1,neh) .* P0_n(2*n,:);
      HOUGH_0_UVZ(3,2*l-1+2*nLG,neh,:) = HOUGH_0_UVZ(3,2*l-1+2*nLG,neh,:) + AUX_0_UVZ(3,2*l-1+2*nLG,neh,:);
    end
  end
end

disp('End of part I (zonal wave number zero)');

disp('Part II');
% PART II ------------------------------------------------------------
% For zonal wave numbers m>0. The frequencies and associated horizontal stucture functions are
% computed as the eigenvalues/eigenvectors of matrices A and B in Swarztrauber and Kasahara (1985).
% Matrices A and B have dimensions 3*N x 3*N because the frequencies are determined in triplets
% corresponding to eastward gravity, westward gravity and westward rotational (rossby) modes.


% Arrays for the Hough functions
HOUGH_UVZ   = zeros(3,M,L,NEH,length(x)); % (complex) Hough functions for m>0
AUX_UVZ     = zeros(3,M,L,NEH,length(x)); % (complex) Auxiliar array for summation (3.22)

% Arrays for the frequencies (eigenvalues)
WEST_G_sy  = zeros(M,nLG/2,NEH);
WEST_G_asy = zeros(M,nLG/2,NEH);
EAST_G_sy  = zeros(M,nLG/2,NEH);
EAST_G_asy = zeros(M,nLG/2,NEH);
WEST_R_sy  = zeros(M,nLR/2,NEH);
WEST_R_asy = zeros(M,nLR/2,NEH);

% Arrays for the coefficients (eigenvectors)
% The coefficients are stored in columns as:
% Symmetric subsystem     ---> (Cm_m,  Am_m,  Bm_m+1, Cm_m+2, Am_m+2, Bm_m+3, ...)
% Antisymmetric subsystem ---> (Bm_m, Cm_m+1, Am_m+1, Bm_m+2, Cm_m+3, Am_m+3, ...)
ABC_WG_sy  = zeros(M,maxN,nLG/2,NEH);
ABC_WG_asy = zeros(M,maxN,nLG/2,NEH);
ABC_EG_sy  = zeros(M,maxN,nLG/2,NEH);
ABC_EG_asy = zeros(M,maxN,nLG/2,NEH);
ABC_WR_sy  = zeros(M,maxN,nLR/2,NEH);
ABC_WR_asy = zeros(M,maxN,nLR/2,NEH);


% calculations
% Matrices A (symmetric subsystem) and B (antisymmetric subsystem)

for m=1:M
  fprintf(1,'  %d of %d\n',m,M);

  % Terms r_n, qm_n and pm_n
  n=m+[0:maxN-1];
  r_n=zeros(maxN,NEH);
  for gl=1:NEH
    for k=1:maxN
      % Eq. (3.26) in Swarztrauber and Kasahara (1985)
      r_n(k,gl)=Ga(gl)*sqrt(n(k).*(n(k)+1));
    end
  end

  pm_n = zeros(maxN,1);
  qm_n = zeros(maxN,1);

  for k=1:maxN
    % Eq. (3.11) in Swarztrauber and Kasahara (1985).
    pm_n(k)=sqrt(((n(k)-1).*(n(k)+1).*(n(k)-m).*(n(k)+m))/(n(k).^2.*(2*n(k)-1).*(2*n(k)+1) ));
    % Eq. (3.11) in Swarztrauber and Kasahara (1985)
    qm_n(k)=m/(n(k).*(n(k)+1));
  end


  % Matrix A is given by eq. (3.28) in Swarztrauber and Kasahara (1985)
  if m==1, disp('  - Matrix A'); end
  %-----------------------------------------------------------------
  A_uud = zeros(maxN,NEH); % Uppermost diagonal elements for Matrix A
  A_ud  = zeros(maxN,NEH); % Upper diagonal elements for Matrix A
  A_md  = zeros(maxN,NEH); % Main diagonal elements for Matrix A
  for k=1:maxN/3
    A_uud(3*k,:)  = pm_n(2*k+1);
    A_ud(3*k-2,:) = r_n(2*k-1,:);
    A_ud(3*k-1,:) = pm_n(2*k);
    A_md(3*k-1,:) = -qm_n(2*k-1);
    A_md(3*k,:)   = -qm_n(2*k);
  end

  A_uud = A_uud(1:maxN-2,:); % Only the first maxN-2 elements are needed.
  A_ud  = A_ud(1:maxN-1,:);  % Only the first maxN-1 elements are needed
  %A_lld=A_uud;              % Lowermost diagonal elements for Matrix A (not needed)
  %A_ld=A_ud;                % Lower diagonal elements for Matrix A (not needed)

  % Creating the pentadiagonal matrix A and calculating it's eigenvalues/eigenvectors.
  S_A  = zeros(maxN,NEH);
  U_A  = zeros(maxN,maxN,NEH);

  for gl=1:NEH
    % Pentadiagonal matrix A
    A = diag(A_uud(:,gl),-2)+diag(A_ud(:,gl),-1)+diag(A_md(:,gl),0)+diag(A_ud(:,gl),1)+diag(A_uud(:,gl),2);

    % The eigenvectors/eigenvalues of A [eq. (3.29) in Swarztrauber and Kasahara (1985)]
    [eig_vec,eig_val]  = eig(A);
    [eig_val,isort]=sort(diag(eig_val));
    eig_vec=eig_vec(:,isort);

    S_A(:,gl)   = eig_val;
    U_A(:,:,gl) = eig_vec;
  end


  % Matrix B is given by eq. (3.31) in Swarztrauber and Kasahara (1985)
  if m==1, disp('  - Matrix B'); end
  %-----------------------------------------------------------------
  B_uud = zeros(maxN,NEH); % Uppermost diagonal elements for Matrix B
  B_ud  = zeros(maxN,NEH); % Upper diagonal elements for Matrix B
  B_md  = zeros(maxN,NEH);   % Main diagonal elements for Matrix B
  for k=1:maxN/3
    B_uud(3*k-2,:) = pm_n(2*k);
    B_ud(3*k-1,:)  = r_n(2*k,:);
    B_ud(3*k,:)    = pm_n(2*k+1);
    B_md(3*k-2,:)  = -qm_n(2*k-1);
    B_md(3*k,:)    = -qm_n(2*k);
  end

  B_uud=B_uud(1:maxN-2,:); % Only the first maxN-2 elements are needed
  B_ud=B_ud(1:maxN-1,:);  % Only the first maxN-1 elements are needed
  %B_lld=B_uud;           % Lowermost diagonal elements for Matrix B (not needed)
  %B_ld=B_ud;             % Lower diagonal elements for Matrix B (not needed)

  % Creating the pentadiagonal matrix A and calculating it's eigenvalues/eigenvectors.
  S_B = zeros(maxN,NEH);
  U_B = zeros(maxN,maxN,NEH);
  for gl=1:NEH
    % Pentadiagonal matrix B.
    B = diag(B_uud(:,gl),-2)+diag(B_ud(:,gl),-1)+diag(B_md(:,gl),0)+diag(B_ud(:,gl),1)+diag(B_uud(:,gl),2);

    % The eigenvectors/eigenvalues of B [eq. (3.32) in Swarztrauber and Kasahara (1985)]
    [eig_vec,eig_val]  = eig(B);
    [eig_val,isort]=sort(diag(eig_val));
    eig_vec=eig_vec(:,isort);

    S_B(:,gl)   = eig_val;
    U_B(:,:,gl) = eig_vec;
  end

  if m==1, disp('  - selecting freqs and coeffs'); end
  % Selecting the frequencies and the coefficients for the vector harmonic expansion.
  % The eigenvalues are classified (for both symmetric and antisymmetric subsystems) as:
  % Westward Gravity -----> (lowest third)
  % Westward Rotational --> (middle third)
  % Eastward Gravity -----> (highest third).

  % Gravity modes
  for la=1:nLG/2
    % The frequencies (eigenvalues)
    WEST_G_sy(m,la,:)    = S_A(N+1-la,:);
    WEST_G_asy(m,la,:)   = S_B(N+1-la,:);
    EAST_G_sy(m,la,:)    = S_A(2*N+la,:);
    EAST_G_asy(m,la,:)   = S_B(2*N+la,:);

    % The coefficients A, B and C (eigenvectors)
    ABC_WG_sy(m,:,la,:)  = U_A(:,N+1-la,:);
    ABC_WG_asy(m,:,la,:) = U_B(:,N+1-la,:);
    ABC_EG_sy(m,:,la,:)  = U_A(:,2*N+la,:);
    ABC_EG_asy(m,:,la,:) = U_B(:,2*N+la,:);
  end

  % Rossby modes
  for la=1:nLR/2
    % The frequencies (eigenvalues)
    WEST_R_asy(m,la,:)   = S_B(N+la,:);
    WEST_R_sy(m,la,:)    = S_A(N+la,:);

    % The coefficients A, B and C (eigenvectors)
    ABC_WR_asy(m,:,la,:) = U_B(:,N+la,:);
    ABC_WR_sy(m,:,la,:)  = U_A(:,N+la,:);
  end


  % HOUGH VECTOR FUNCTIONS ===========================================
  %  Normalized Associated Legendre functions (Pm_n) -----------------
  %  The associated Legendre functions are evaluated for degrees n=0,...,2*N and orders M = 0 and 1,for
  %  each element of x. N must be a scalar integer and X must contain real values between -1 <= x <= 1.

  % Deffining the array for the Associated Legendre functions (normalized)
  Pm_n    = zeros(2*N,length(x));
  Pmm1_n  = zeros(2*N,length(x));
  PmM1_n  = zeros(2*N,length(x));
  Pmm1_n1 = zeros(2*N,length(x));
  PmM1_n1 = zeros(2*N,length(x));


  if m==1, disp('  - Associated Legendre Functions'); end
  for n=m:2*N-1+m
    aux = legendre(n,x,'norm');
    Pm_n(n+1-m,:) = aux(m+1,:);       % P(m,n)

    Pmm1_n(n+1-m,:) = aux(m,:);       % P(m-1,n)

    if n>=m+1
      PmM1_n(n+1-m,:) = aux(m+2,:);   % P(m+1,n)
    end

    aux1 = legendre(n-1,x,'norm');
    Pmm1_n1(n+1-m,:) = aux1(m,:);     % P(m-1,n-1)

    if n-1>=m+1
      PmM1_n1(n+1-m,:) = aux1(m+2,:); % P(m+1,n-1)
    end

  end

  % Derivative of associated Legendre functions with respect to latitude (eq. (3.3))
  dPm_n_dLat=zeros(2*N,length(x));
  for n=m:2*N-1+m
    dPm_n_dLat(n+1-m,:) = (1/2)*(sqrt((n-m)*(n+m+1))*PmM1_n(n+1-m,:)-sqrt((n+m)*(n-m+1))*Pmm1_n(n+1-m,:));
  end

  % The term given by eq. (3.4)
  mPm_n_cosLat=zeros(2*N,length(x));
  for n=m:2*N-1+m
    mPm_n_cosLat(n+1-m,:) = (1/2)*sqrt((2*n+1)/(2*n-1))*(sqrt((n+m)*(n+m-1))*Pmm1_n1(n+1-m,:)+sqrt((n-m)*(n-m-1))*PmM1_n1(n+1-m,:));
  end


  % The spherical vector harmonics -----------------------------------
  if m==1, disp('  - spherical vector harmonics'); end
  % The spherical vector harmonics will be computed using eqs. (3.1) without the factor e^(i m lambda), since
  % this factor will be canceled in the summation (3.22)
  y1m_n = zeros(3,2*N,length(x)); % (complex)
  y2m_n = zeros(3,2*N,length(x)); % (complex)
  y3m_n = zeros(3,2*N,length(x)); % (complex)

  y3m_n(3,:,:) = Pm_n;
  for n=m:2*N-1+m
    y1m_n(1,n+1-m,:) = I.*mPm_n_cosLat(n+1-m,:)./sqrt(n*(n+1));
    y1m_n(2,n+1-m,:) = dPm_n_dLat(n+1-m,:)./sqrt(n*(n+1));
  end
  y2m_n(1,:,:) = -y1m_n(2,:,:);
  y2m_n(2,:,:) = y1m_n(1,:,:);


  % HOUGH vector functions -----------------------------------------
  % The HOUGH vector functions are computed using eq. (3.22) in Swarztrauber and Kasahara (1985)

  if m==1, disp('  - HVF: gravity'); end

  for l=1:nLG/2
    for neh=1:NEH
      for n=1:N
        % ======================== SYMMETRIC SUBSISTEMS ======================
        % ------------------------- Westward gravity -------------------------
        AUX_UVZ(1,m,2*l-1,neh,:)   = I*ABC_WG_sy(m,3*n-1,l,neh) .* y1m_n(1,2*n-1,:) + ABC_WG_sy(m,3*n,l,neh) .* y2m_n(1,2*n,:);
        HOUGH_UVZ(1,m,2*l-1,neh,:) = HOUGH_UVZ(1,m,2*l-1,neh,:) + AUX_UVZ(1,m,2*l-1,neh,:);
        AUX_UVZ(2,m,2*l-1,neh,:)   = I*ABC_WG_sy(m,3*n-1,l,neh) .* y1m_n(2,2*n-1,:) + ABC_WG_sy(m,3*n,l,neh) .* y2m_n(2,2*n,:);
        HOUGH_UVZ(2,m,2*l-1,neh,:) = HOUGH_UVZ(2,m,2*l-1,neh,:) + AUX_UVZ(2,m,2*l-1,neh,:);
        AUX_UVZ(3,m,2*l-1,neh,:)   = -ABC_WG_sy(m,3*n-2,l,neh) .* y3m_n(3,2*n-1,:);
        HOUGH_UVZ(3,m,2*l-1,neh,:) = HOUGH_UVZ(3,m,2*l-1,neh,:) + AUX_UVZ(3,m,2*l-1,neh,:);
        % ------------------------- Eastward gravity -------------------------
        AUX_UVZ(1,m,2*l-1+nLG,neh,:)   = I*ABC_EG_sy(m,3*n-1,l,neh) .* y1m_n(1,2*n-1,:) + ABC_EG_sy(m,3*n,l,neh) .* y2m_n(1,2*n,:);
        HOUGH_UVZ(1,m,2*l-1+nLG,neh,:) = HOUGH_UVZ(1,m,2*l-1+nLG,neh,:) + AUX_UVZ(1,m,2*l-1+nLG,neh,:);
        AUX_UVZ(2,m,2*l-1+nLG,neh,:)   = I*ABC_EG_sy(m,3*n-1,l,neh) .* y1m_n(2,2*n-1,:) + ABC_EG_sy(m,3*n,l,neh) .* y2m_n(2,2*n,:);
        HOUGH_UVZ(2,m,2*l-1+nLG,neh,:) = HOUGH_UVZ(2,m,2*l-1+nLG,neh,:) + AUX_UVZ(2,m,2*l-1+nLG,neh,:);
        AUX_UVZ(3,m,2*l-1+nLG,neh,:)   = -ABC_EG_sy(m,3*n-2,l,neh) .* y3m_n(3,2*n-1,:);
        HOUGH_UVZ(3,m,2*l-1+nLG,neh,:) = HOUGH_UVZ(3,m,2*l-1+nLG,neh,:) + AUX_UVZ(3,m,2*l-1+nLG,neh,:);
        % ======================== ANTISYMMETRIC SUBSISTEMS ==================
        % ------------------------- Westward gravity -------------------------
        AUX_UVZ(1,m,2*l,neh,:)   = I*ABC_WG_asy(m,3*n,l,neh) .* y1m_n(1,2*n,:) + ABC_WG_asy(m,3*n-2,l,neh) .* y2m_n(1,2*n-1,:);
        HOUGH_UVZ(1,m,2*l,neh,:) = HOUGH_UVZ(1,m,2*l,neh,:) + AUX_UVZ(1,m,2*l,neh,:);
        AUX_UVZ(2,m,2*l,neh,:)   = I*ABC_WG_asy(m,3*n,l,neh) .* y1m_n(2,2*n,:) + ABC_WG_asy(m,3*n-2,l,neh) .* y2m_n(2,2*n-1,:);
        HOUGH_UVZ(2,m,2*l,neh,:) = HOUGH_UVZ(2,m,2*l,neh,:) + AUX_UVZ(2,m,2*l,neh,:);
        AUX_UVZ(3,m,2*l,neh,:)   = -ABC_WG_asy(m,3*n-1,l,neh) .* y3m_n(3,2*n,:);
        HOUGH_UVZ(3,m,2*l,neh,:) = HOUGH_UVZ(3,m,2*l,neh,:) + AUX_UVZ(3,m,2*l,neh,:);
        % ------------------------- Eastward gravity -------------------------
        AUX_UVZ(1,m,2*l+nLG,neh,:)   = I*ABC_EG_asy(m,3*n,l,neh) .* y1m_n(1,2*n,:) + ABC_EG_asy(m,3*n-2,l,neh) .* y2m_n(1,2*n-1,:);
        HOUGH_UVZ(1,m,2*l+nLG,neh,:) = HOUGH_UVZ(1,m,2*l+nLG,neh,:) + AUX_UVZ(1,m,2*l+nLG,neh,:);
        AUX_UVZ(2,m,2*l+nLG,neh,:)   = I*ABC_EG_asy(m,3*n,l,neh) .* y1m_n(2,2*n,:) + ABC_EG_asy(m,3*n-2,l,neh) .* y2m_n(2,2*n-1,:);
        HOUGH_UVZ(2,m,2*l+nLG,neh,:) = HOUGH_UVZ(2,m,2*l+nLG,neh,:) + AUX_UVZ(2,m,2*l+nLG,neh,:);
        AUX_UVZ(3,m,2*l+nLG,neh,:)   = -ABC_EG_asy(m,3*n-1,l,neh) .* y3m_n(3,2*n,:);
        HOUGH_UVZ(3,m,2*l+nLG,neh,:) = HOUGH_UVZ(3,m,2*l+nLG,neh,:) + AUX_UVZ(3,m,2*l+nLG,neh,:);
      end
    end
  end

  if m==1, disp('  - HVF: rossby'); end

  for l=1:nLR/2
    for neh=1:NEH
      for n=1:N
        % ======================== SYMMETRIC SUBSISTEMS ======================
        % ------------------------- Westward rossby --------------------------
        AUX_UVZ(1,m,2*l+2*nLG,neh,:)   = I*ABC_WR_sy(m,3*n-1,l,neh) .* y1m_n(1,2*n-1,:) + ABC_WR_sy(m,3*n,l,neh) .* y2m_n(1,2*n,:);
        HOUGH_UVZ(1,m,2*l+2*nLG,neh,:) = HOUGH_UVZ(1,m,2*l+2*nLG,neh,:) + AUX_UVZ(1,m,2*l+2*nLG,neh,:);
        AUX_UVZ(2,m,2*l+2*nLG,neh,:)   = I*ABC_WR_sy(m,3*n-1,l,neh) .* y1m_n(2,2*n-1,:) + ABC_WR_sy(m,3*n,l,neh) .* y2m_n(2,2*n,:);
        HOUGH_UVZ(2,m,2*l+2*nLG,neh,:) = HOUGH_UVZ(2,m,2*l+2*nLG,neh,:) + AUX_UVZ(2,m,2*l+2*nLG,neh,:);
        AUX_UVZ(3,m,2*l+2*nLG,neh,:)   = -ABC_WR_sy(m,3*n-2,l,neh) .* y3m_n(3,2*n-1,:);
        HOUGH_UVZ(3,m,2*l+2*nLG,neh,:) = HOUGH_UVZ(3,m,2*l+2*nLG,neh,:) + AUX_UVZ(3,m,2*l+2*nLG,neh,:);
        % ======================== ANTISYMMETRIC SUBSISTEMS ==================
        % ------------------------- Westward rossby --------------------------
        AUX_UVZ(1,m,2*l-1+2*nLG,neh,:)   = I*ABC_WR_asy(m,3*n,l,neh) .* y1m_n(1,2*n,:) + ABC_WR_asy(m,3*n-2,l,neh) .* y2m_n(1,2*n-1,:);
        HOUGH_UVZ(1,m,2*l-1+2*nLG,neh,:) = HOUGH_UVZ(1,m,2*l-1+2*nLG,neh,:) + AUX_UVZ(1,m,2*l-1+2*nLG,neh,:);
        AUX_UVZ(2,m,2*l-1+2*nLG,neh,:)   = I*ABC_WR_asy(m,3*n,l,neh) .* y1m_n(2,2*n,:) + ABC_WR_asy(m,3*n-2,l,neh) .* y2m_n(2,2*n-1,:);
        HOUGH_UVZ(2,m,2*l-1+2*nLG,neh,:) = HOUGH_UVZ(2,m,2*l-1+2*nLG,neh,:) + AUX_UVZ(2,m,2*l-1+2*nLG,neh,:);
        AUX_UVZ(3,m,2*l-1+2*nLG,neh,:)   = -ABC_WR_asy(m,3*n-1,l,neh) .* y3m_n(3,2*n,:);
        HOUGH_UVZ(3,m,2*l-1+2*nLG,neh,:) = HOUGH_UVZ(3,m,2*l-1+2*nLG,neh,:) + AUX_UVZ(3,m,2*l-1+2*nLG,neh,:);
      end
    end
  end

end  % End the zonal wave numbers
disp('End of part II (zonal wave numbers m>0)');

% The first symmetric (lowest order) eastward gravity mode is eigenvector of matrix E
WEST_R_0_sy           = Sa_E(2:nLR/2+1,:);

WEST_R_0_asy(2:end,:) = WEST_R_0_asy(1:end-1,:);
WEST_R_0_asy(1,:)     = S_C(1,:);

EAST_G_0_sy(1,:)      = Sa_E(1,:);

% end of calculations

% concatenate frequencies (eigenvalues):
% part 1 (zonal mean):
FREQS_0=zeros(L,NEH);
% Gravity modes
FREQS_0(1:2:nLG,:)       = WEST_G_0_sy;
FREQS_0(2:2:nLG,:)       = WEST_G_0_asy;
FREQS_0(nLG+1:2:2*nLG,:) = EAST_G_0_sy;
FREQS_0(nLG+2:2:2*nLG,:) = EAST_G_0_asy;
% Rossby modes
FREQS_0(2*nLG+1:2:L,:) = WEST_R_0_asy;
FREQS_0(2*nLG+2:2:L,:) = WEST_R_0_sy;

% part 2 (eddies):
FREQS_m = zeros(M,L,NEH);
% Gravity modes
FREQS_m(:,1:2:nLG,:)       = WEST_G_sy;
FREQS_m(:,2:2:nLG,:)       = WEST_G_asy;
FREQS_m(:,nLG+1:2:2*nLG,:) = EAST_G_sy;
FREQS_m(:,nLG+2:2:2*nLG,:) = EAST_G_asy;
% Rossby modes
FREQS_m(:,2*nLG+1:2:L,:) = WEST_R_asy;
FREQS_m(:,2*nLG+2:2:L,:) = WEST_R_sy;

out=struct;
out.HOUGH_UVZ   = HOUGH_UVZ;
out.HOUGH_0_UVZ = HOUGH_0_UVZ;
out.FREQS_0     = FREQS_0;
out.FREQS_m     = FREQS_m;
end


function [C,S_C,U_C]=calc_CD(p_n,r_n,add)
  %
  % add=0  --> C
  % add=1  --> D
  %
  % C, Eq. (4.8) in Swarztrauber and Kasahara (1985)
  % D, Eq. (4.11) in Swarztrauber and Kasahara (1985)
  %

  [nn,NEH]=size(r_n);
  N=nn/2;

  C_dd = zeros(N-1,NEH);
  C_md = zeros(N,NEH);

  % Upper (and lower) diagonal elements:
  for k=1:N-1
    C_dd(k,:) = p_n(2*k-1+add) * p_n(2*k+add);
  end

  if add==0
    % First index for r_n is zero while for p_n is one.
    C_md(1,:) = r_n(1,:) .* r_n(1,:) + p_n(1) * p_n(1);
    i0=2;
  else
    i0=1;
  end

  % Main diagonal elements
  for k=i0:N
    C_md(k,:) = r_n(2*k-1+add,:).*r_n(2*k-1+add,:) + p_n(2*k-2+add).*p_n(2*k-2+add) + p_n(2*k-1+add).*p_n(2*k-1+add);
  end

  % Creating the tridiagonal matrix and calculating it's eigenvalues/eigenvectors.
  C       = zeros(N,N,NEH);
  S_C     = zeros(N,NEH);
  S_Caux  = zeros(N,NEH);
  U_C     = zeros(N,N,NEH);

  for gi=1:NEH
    % Tridiagonal matrix
    C(:,:,gi) = diag(C_dd(:,gi),-1) + diag(C_md(:,gi),0) + diag(C_dd(:,gi),1);
    % Eigenvalues/eigenvectors of C/D [Eq. (4.9/4.12) in Swarztrauber and Kasahara (1985)]
    [U_C(:,:,gi),aux] = eig(squeeze(C(:,:,gi)));
    [S_Caux(:,gi) IC2sort] = sort(diag(squeeze(aux)));   % Sorts eigenvalues and retrieves their indices.
    U_C(:,:,gi) = U_C(:,IC2sort,gi);                     % Corresponding eigenvectors sorted
  end

  S_C = sqrt(S_Caux);   % Frequencies (Sigma)
end
