function varargout=vertical_structure(Tprof,Plev,varargin)
%  Solves the vertical structure equation, returning the vertical
%  structure functions and equivalent heights
%
%  Tprof, reference temperature (mean vertical profile, K)
%  Plev, corresponding pressure levels (Pa)
%  varargin:
%    ws0, If true (default false) the pressure vertical velocity  is zero at surface.
%    n_leg, number of Legendre polynomials, length(Plev)+20 by default
%    nk, n functions to keep (cannot exceed the number of pressure levels, the default)
%    save, create file [True]
%    format, file format: [nc] or mat
%    attrs, attributes to save [{}]
%    label, start of the saved filename ['out']
%
%  Output:
%    Gn, hk or Gn, hk and output filename if save is true
%
%  Example:
%    a=load('tprof.mat');
%
%    % netcdf output:
%    [Gn,hk,fname]=vertical_structure(a.Tprof,a.Plev,'save',1,'ws0',true);
%
%    nc=netcdf.open(fname)
%    i=netcdf.inqVarID(nc,'plev');
%    plev=netcdf.getVar(nc,i);
%    semilogy(Gn(5,:),plev,'r.')
%
%    % mat output:
%    [Gn,hk,fname]=vertical_structure(Tprof,Plev,'save',1,'ws0',true,'format','mat');
%
%    b=load(fname);
%    plot(b.hk)

constants;

ws0   = false;
n_leg = 'auto';
nk    = length(Plev);
save  = true;

for i=1:length(varargin)
  if isequal(varargin{i},'ws0')
    ws0=varargin{i+1};
  elseif isequal(varargin{i},'n_leg')
    n_leg=varargin{i+1};
  elseif isequal(varargin{i},'nk')
    nk=varargin{i+1};
  elseif isequal(varargin{i},'save')
    save=varargin{i+1};
  end
end

if isequal(n_leg,'auto'),  n_leg=length(Plev)+20; end

Tprof=double(Tprof);
Plev=double(Plev);

J=n_leg;
GL=2*J-1;  % number of Gaussian levels

% Gaussian levels (i.e. points (Gp)) and Gaussian weights (Gw)
[Gp,Gw] = lgwt(GL,-1,1);
%  Gp=flipud(Gp);

% Cubic spline interpolation of reference temperature from pressure to sigma levels
Plev_s  = (Gp+1)*const.ps/2; % pressure levels that correspond to the chosen Gaussian sigma levels
Tprof_s = spline(Plev, Tprof,Plev_s);

% Reference Temperature at sigma=1, linearly extrapolated:
Gp1 = 1;
Tprof_s1 = Tprof_s(2) + (Gp1-Gp(2))/(Gp(1)-Gp(2)) * (Tprof_s(1)-Tprof_s(2));

% Static stability in the sigma system:
Gamma0=stability(Tprof_s,Gp);

% Matrix Mij (Eq. (A12) in Kasahara (1984))
[M,P_s]=calc_M(n_leg,Gp,Gw,Gamma0,Tprof_s1,ws0);

% Eigenvectors and eigenvalues of matrix Mij
% Note that using eig(M), the eigenvalues/eigenvectors are not necessarily ordered.
% But since matrix M is symmetric, the eigenvalues are also singular values and
% therefore can be obtained with the svd function.
useSvd=false;
if useSvd
  [V,S,E]=svd(M);  % With svd the eigenvalues/eigenvectors are ordered
  S=flipud(diag(S));
else
  [V,S]=eig(M);
  [S,i]=sort(diag(S));
  V=V(:,i);
  V=fliplr(V);
end

if ws0 & S(1)~=0.
  disp('Warning: forcing S(1) to 0');
  S(1)=0;
end

% Eigenfunctions (Gn), i.e. the Vertical Structure Functions (Eq. (A8) in Kasahara (1984))
Gn_all=V'*P_s';

% Re-ordering the eigenfunctions (as k=0,1,...,J-1) and taking the 1st nkMax eigenfunctions:
nkMax=length(Plev);
Gn=flipud(Gn_all);
Gn=Gn(1:nkMax,:);

% The equivalent heights re-ordered
cond=S==0;
S(cond)=1;
hk=const.H00./S;
hk(cond)=inf;
hk=hk(1:nkMax);

varargout{1}=Gn(1:nk,:);
varargout{2}=hk(1:nk);
if save
  data=struct;
  data.Gn     = Gn;
  data.hk     = hk;
  data.gamma0 = Gamma0;
  data.plev   = Plev_s;
  data.plev0  = Plev;
  data.tprof0 = Tprof;
  fsave=save_out(data,ws0,n_leg,varargin{:});
  varargout{3}=fsave;
end

end


function s=stability(Tref,Gp)
  %
  % Static stability in the sigma system (eq. (A3) in Kasahara (1984))
  % Derivative, by finite differences, of reference temperature (Tref) with respect to logarithm of p/ps (dT0_dLn_pps).
  %

  constants;

  pps = (Gp+1)/2; % p/Ps
  dTref_dLn_pps = zeros(size(Gp));

  % delta s:
  Ds = diff(log(pps));

  % forward differences (1st order):
  dTref_dLn_pps(1) = (Tref(2)- Tref(1)) / Ds(1);


  % Centred differences (2st order)
  for k=2:length(Gp)-1
    dTref_dLn_pps(k) = (1/(Ds(k-1)*Ds(k)*(Ds(k-1)+Ds(k)))) .* (Ds(k-1)^2*Tref(k+1)-Ds(k)^2*Tref(k-1)-(Ds(k-1)^2-Ds(k)^2)*Tref(k));
  end

  % Backward differences (1st order)
  dTref_dLn_pps(end) = (Tref(end)-Tref(end-1)) / Ds(end);

  % The static stability in the sigma system (Gamma0)
  s=(const.Qsi*Tref)./(1+Gp)-0.5./pps.*dTref_dLn_pps;
end


function [M,P_s]=calc_M(J,Gp,Gw,Gamma0,Tref1,ws0)
  %
  %  Matrix Mij (Eq. (A12) in Kasahara (1984))
  %

  constants;

  GL=2*J-1;

  % Normalized Associated Legendre functions
  %m = 0 % order
  %uP_s  = zeros(GL,J); % Unnormalized
  P_s   = zeros(GL,J); % Normalized
  P_s1  = zeros(J,1);  % Normalized (at sigma=1)


  for j=1:J
    if j-1==0
      aux = legendre(j-1,Gp,'norm');
      P_s(:,j)=aux';
    else
      aux = legendre(j-1,Gp,'norm');
      P_s(:,j)=aux(1,:);
    end
  end

  % Legendre polynomials at sigma=1
  for j=1:J
    aux = legendre(j-1,1,'norm');
    P_s1(j)=aux(1);
  end

  % Derivative of Legendre polynomials with respect to sigma (d_P_ds)
  d_P_ds = zeros(GL,J);
  % The derivative of P_s(j=zero)=0 (all sigmas). Therefore, the index j of d_P_ds starts at j=1,
  % where the derivative of P_s(one) (all sigmas) is stored, and so on.
  for j=2:J
    d_P_ds(:,j) = (j-1)*Gp./(Gp.^2-1).*P_s(:,j) - (j-1)./(Gp.^2-1)*sqrt((2.*(j-1)+1)/(2.*(j-1)-1)) .* P_s(:,j-1);
  end

  % Matrix Mij (Eq. (A12) in Kasahara (1984))
  M  = zeros(J,J);   % Initializing matrix Mij
  for i=1:j
    for j=1:j
      if ws0 % w=0 at surface
        M(i,j) = const.T00 * sum(((Gp+1.0)./Gamma0.*d_P_ds(:,i).*d_P_ds(:,j)).*Gw);
      else
        M(i,j) = const.T00 * sum(((Gp+1.0)./Gamma0.*d_P_ds(:,i).*d_P_ds(:,j)).*Gw) + const.T00 * (2.0/Tref1*P_s1(i).*P_s1(j));
      end
    end
  end

end


function fsave=save_out(data,ws0,n_leg,varargin)
  label='out';
  format='nc';
  attrs=struct;

  for i=1:length(varargin)
    if isequal(varargin{i},'label')
      label=varargin{i+1};
    elseif isequal(varargin{i},'format')
      format=varargin{i+1};
    elseif isequal(varargin{i},'attrs')
      attrs=varargin{i+1};
    end
  end

  if ws0, ws0='True';
  else,   ws0='False';
  end

  attrs.ws0         = ws0;
  attrs.n_leg       = int8(n_leg);
  attrs.platform    = computer;
  attrs.environment = 'matlab';
  attrs.version     = version;

  fsave=sprintf('%s_vs_ws0%s.%s',label,string(ws0),format);

  fprintf(1,'saving %s\n',fsave);
  if isequal(format,'mat')
    % update data with attrs:
    f = fieldnames(attrs);
    for i = 1:length(f)
      data.(f{i})=attrs.(f{i});
    end
    save(fsave, '-struct','data');
  elseif isequal(format,'nc')
    save_nc(fsave,data,attrs);
  else
    disp('Unknown format, use nc or mat');
  end

end


function save_nc(fname,data,attrs)
  nc=netcdf.create(fname,'CLOBBER');

  % dimensions:
  [nkmax,GL]=size(data.Gn);
  dim_nk=netcdf.defDim(nc,'nk_max',nkmax);
  dim_gl=netcdf.defDim(nc,'GL',GL); % Gaussian levels

  % dimensions for original variables:
  dim_p0=netcdf.defDim(nc,'nlevels0',length(data.plev0));


  % variables:
  var_gn = netcdf.defVar(nc,'Gn',    'NC_DOUBLE',[dim_gl,dim_nk]); % dimnames are swapped
  netcdf.putAtt(nc,var_gn,'long_name','Vertical structure functions');

  var_hk = netcdf.defVar(nc,'hk',    'NC_DOUBLE',dim_nk);
  netcdf.putAtt(nc,var_hk,'long_name','Equivalent heights');

  var_g0 = netcdf.defVar(nc,'gamma0','NC_DOUBLE',dim_gl);
  netcdf.putAtt(nc,var_g0,'long_name','Static stability in the sigma system');

  var_pl = netcdf.defVar(nc,'plev',  'NC_DOUBLE',dim_gl);
  netcdf.putAtt(nc,var_pl,'long_name','Pressure levels corresponding to the gaussian sigma levels');
  netcdf.putAtt(nc,var_pl,'units','Pa');

  % original variables:
  var_p0 = netcdf.defVar(nc,'plev0',  'NC_DOUBLE',dim_p0);
  netcdf.putAtt(nc,var_p0,'long_name','Original pressure levels');
  netcdf.putAtt(nc,var_p0,'units','Pa');

  var_t0 = netcdf.defVar(nc,'tprof0',  'NC_DOUBLE',dim_p0);
  netcdf.putAtt(nc,var_t0,'long_name','Original temperature profile');
  netcdf.putAtt(nc,var_t0,'units','K');

  % global attributes:
  vid = netcdf.getConstant('GLOBAL');
  netcdf.putAtt(nc,vid,'date',datestr(now));
  f=fieldnames(attrs);
  for i=1:length(f)
    netcdf.putAtt(nc,vid,f{i},attrs.(f{i}));
  end

  netcdf.endDef(nc);

  % fill variables:
  netcdf.putVar(nc,var_gn,data.Gn');
  netcdf.putVar(nc,var_hk,data.hk);
  netcdf.putVar(nc,var_g0,data.gamma0);
  netcdf.putVar(nc,var_pl,data.plev);
  netcdf.putVar(nc,var_p0,data.plev0);
  netcdf.putVar(nc,var_t0,data.tprof0);

  netcdf.close(nc);
end
