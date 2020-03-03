function out=hvf_barotropic(nLR,nLG,M,trunc,x)
%  Compute the Hough vector functions as described in the paper of Swarztrauber and Kasahara (1985).
%  Limiting case of hk[0] = inf (The Haurwitz waves).
%
%  Part I: The frequencies and the Hough functions are computed for zonal wave number m = 0.
%  Part II: The frequencies and the Hough functions are computed for zonal wave numbers m > 0.
%
%  See hough_functions

constants;

fprintf(1,'\n- HVF barotropic -\n')

N=trunc; % truncation order
NEH=1;
L = nLR+2*nLG; % Total number of meridional modes used in the expansion (should be even)

% Total number of eigenvalues/eigenvectors for matrices A and B.
maxN = 3*N;

% Dimensions --------------------
% Arrays fo the Hough functions
HOUGH_0_UVZ = zeros(3,L,length(x)); % Hough functions for n=0


disp('Part I');
% PART I ----------------------------------------------------------
% For zonal wave number m=0.

% HOUGH VECTOR FUNCTIONS ============================================
%  Normalized Associated Legendre Functions (Pm_n) ------------------
% Computes the associated Legendre functions of degree N and order M = 0, 1, ..., N,
% evaluated for each element of X. N must be a scalar integer and X must contain real values between
%  -1 <= X <= 1.

% P(1,n)
P1_n = zeros(2*N,length(x));
%P1_n(1,:) = 0;   % n=0 => P(1,0)=0
for n=1:2*N-1
  aux = legendre(n,x,'norm');
  P1_n(n+1,:) = aux(2,:); % P(1,n)
end

% P(0,n+1)
P0_nM1 = zeros(2*N,length(x));
for n=0:2*N-1
  aux = legendre(n+1,x,'norm');
  P0_nM1(n+1,:) = aux(1,:); % P(0,n+1)
end

% P(0,n-1)
P0_nm1 = zeros(2*N,length(x));
%P0_nm1(1,:) = 0;   % n=0 => P(0,-1)=0
%P0_nm1(2,:) = 0;   % n=1 => P(0,0)=Const=0. We set the constant to zero because we are
                    % dealing with geopotential perturbations (i.e. with zero mean)

for n=2:2*N-1
  aux = legendre(n-1,x,'norm');
  P0_nm1(n+1,:) = aux(1,:); % P(0,n-1)
end

p0_n=zeros(2*N,1);
p0_n1=zeros(2*N,1);
for n=1:2*N
  % From eq. (3.11) in Swarztrauber and Kasahara (1985).
  p0_n(n)  = sqrt( ((n+1)) / (n*(2*n-1).*(2*n+1)) );       % P(0,n)/sqrt(n(n-1))
  p0_n1(n) = sqrt( (    n) / ((n+1)*(2*n+1).*(2*n+3) ));   % P(0,n+1)/sqrt((n+1)(n+2))
end

% Replicate to have dimensions (2*N x nLat)
p0_nMAT  = repmat(p0_n,[1 length(x)]);
p0_n1MAT = repmat(p0_n1,[1 length(x)]);


% HOUGH vector functions -------------------------------------------
% The HOUGH vector functions are computed using eq. (3.22) in Swarztrauber and Kasahara (1985)

% Rotational (ROSSBY) MODES
HOUGH_0_UVZ(1,2*nLG+1:end,:) = - P1_n(1:nLR,:);   % Eq. (5.1)

% Eq. 5.13
HOUGH_0_UVZ(3,2*nLG+1:end,:) =  (2*const.Er*const.Om)/sqrt(const.g) * (p0_nMAT(1:nLR,:).* P0_nm1(1:nLR,:) + p0_n1MAT(1:nLR,:).*P0_nM1(1:nLR,:));
% Note: The third component was multiplied by sqrt(g) in order to use the same algorithm in the reconstruction as
% that used with dimensionalised variables, by setting artificially the barotropic equivalent height (which is infinity) to one.


% GRAVITY MODES
% These modes are all zero

disp('End of part I (zonal wave number zero)');


disp('Part II');
% For zonal wave numbers m>0. The frequencies and associated horizontal stucture functions are
% computed as the eigenvectors/eigenvalues of matrices A and B in Swarztrauber and Kasahara (1985).
% Matrices A and B have dimensions 3*N x 3*N because the frequencies are determined in triplets
% corresponding to eastward gravity, westward gravity and westward rotational (rossby) modes.

% Dimensions --------------------
% Arrays for the Hough functions
HOUGH_UVZ = zeros(3,M,L,length(x)); % (complex) Hough functions for m>0


for m=1:M  % Start the zonal wave numbers
  % HOUGH VECTOR FUNCTIONS
  % The normalized Legendre functions (Pm_n)

  Pm_n    = zeros(2*N,length(x));
  Pmm1_n  = zeros(2*N,length(x));
  PmM1_n  = zeros(2*N,length(x));
  Pmm1_n1 = zeros(2*N,length(x));
  PmM1_n1 = zeros(2*N,length(x));
  Pm_nm1  = zeros(2*N,length(x));
  Pm_nM1  = zeros(2*N,length(x));

  for n=m:2*N-1+m
    aux = legendre(n,x,'norm');

    Pm_n(n+1-m,:)   = aux(m+1,:);     % P(m,n)
    Pmm1_n(n+1-m,:) = aux(m,:);       % P(m-1,n)

    if n>=m+1
      PmM1_n(n+1-m,:) = aux(m+2,:);   % P(m+1,n)
    end

    aux1 = legendre(n-1,x,'norm');

    Pmm1_n1(n+1-m,:) = aux1(m,:);     % P(m-1,n-1)

    if n-1>=m+1
      PmM1_n1(n+1-m,:) = aux1(m+2,:); % P(m+1,n-1)
    end

    if n-1>=m
      Pm_nm1(n+1-m,:) = aux1(m+1,:);  % P(m,n-1)
    end

    aux = legendre(n+1,x,'norm');
    Pm_nM1(n+1-m,:) = aux(m+1,:);     % P(m,n+1)
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


  % The spherical vector harmonics
  % - Will be computed using eqs. (3.1) without the factor e^(i m lambda), since
  % this factor will be canceled in the summation (3.22).
  y2m_n = zeros(3,2*N,length(x)); % complex
  y1m_n = zeros(3,2*N,length(x)); % complex

  y3m_nm1 = zeros(3,2*N,length(x)); % complex
  y3m_nM1 = zeros(3,2*N,length(x)); % complex

  y3m_nm1(3,:,:) = Pm_nm1;
  y3m_nM1(3,:,:) = Pm_nM1;
  for n=m:2*N-1+m
    y1m_n(1,n+1-m,:) = const.I.*mPm_n_cosLat(n+1-m,:)./sqrt(n*(n+1));
    y1m_n(2,n+1-m,:) = dPm_n_dLat(n+1-m,:)./sqrt(n*(n+1));
  end
  y2m_n(1,:,:) = -y1m_n(2,:,:);
  y2m_n(2,:,:) = y1m_n(1,:,:);

  pm_n  = zeros(2*N,length(x));
  pm_n1 = zeros(2*N,length(x));
  for n=m:2*N-1+m
    % From eq. (3.11) in Swarztrauber and Kasahara (1985)
    pm_n(n+1-m,:)  = sqrt( ((n+1).*(n-m).*(n+m)) / (n.^3.*(2*n-1).*(2*n+1)) );       % P(m,n)/sqrt(n(n-1))
    pm_n1(n+1-m,:) = sqrt( ((n).*(n-m+1).*(n+m+1))/((n+1).^3.*(2*n+1).*(2*n+3) ));   % P(m,n+1)/sqrt((n+1)(n+2))
  end

  % HOUGH vector functions -----------------------------------------
  % The HOUGH vector functions are computed using eq. (3.22) in Swarztrauber and Kasahara (1985)

  % Rotational (ROSSBY) MODES --------------------------------------
  HOUGH_UVZ(1,m,2*nLG+1:end,:) = y2m_n(1,1:nLR,:);
  HOUGH_UVZ(2,m,2*nLG+1:end,:) = y2m_n(2,1:nLR,:);
  HOUGH_UVZ(3,m,2*nLG+1:end,:) = (2*const.Er*const.Om)/sqrt(const.g) * (pm_n(1:nLR,:).*squeeze(y3m_nm1(3,1:nLR,:)) + pm_n1(1:nLR,:).*squeeze(y3m_nM1(3,1:nLR,:)));

  % Note: The third component was multiplied by sqrt(g) in order to use the same algorithm in the reconstruction as
  % that used with dimensionalised variables, by setting artificially the barotropic equivalent height (which is infinity) to one.

  % GRAVITY MODES
  % These modes are all zero

end  % End the zonal wave numbers

% Arrays for the frequencies (eigenvalues)
SIGMAS = zeros(M,nLR);
S_auxS = zeros(M,maxN);

for m=1:M
  for n=m:maxN
    S_auxS(m,n) = -m/(n*(n+1));
  end
  SIGMAS(m,:) = S_auxS(m,m:nLR+m-1);
end

disp('End of part II (zonal wave numbers m>0)');

out=struct;
out.HOUGH_UVZ        = HOUGH_UVZ;
out.HOUGH_0_UVZ      = HOUGH_0_UVZ;
out.SIGMAS           = SIGMAS;

end
