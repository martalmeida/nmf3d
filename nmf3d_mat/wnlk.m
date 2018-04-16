function varargout=wnlk(vfile,hfile,data,varargin)
%  Compute the Vertical, Fourier and Hough transforms of zonal and
%  meridional wind, and geopotential perturbation (from the reference
%  geopotential), i.e. the 3-D spectrum of total energy w_nlk.
%
%  vfile: equivalent heights and vertical structure functions
%  hfile: Hough functions
%  data: u,v,phi data
%  varargin:
%    save, create file [true]
%    format, file format: [nc] or npz
%    attrs, attributes to save [{}]
%    label, start of the saved filename ['out']
%
%  Returns the expansion coefficients (w_nlk) and the zonal expansion
%    coefficients (w_0lk) as well and saved filename of save is true


constants;

save=true; % create file
for i=1:length(varargin)
  if isequal(varargin{i},'save')
    save=varargin{i+1};
  end
end

% check number of hfiles:
try
  hfile_B=hfile{2};
  hfile=hfile{1};
end

fprintf(1,' - loading parameters from Hough functions file:\n    %s\n',hfile);
if endsWith(hfile,'.mat')
  ohfile=load(hfile);

  [max_zonal_wave_number,quarter_number_gravitical_modes, number_equivalent_heights]=size(ohfile.WEST_G_sy);
  half_number_rossby_modes=size(ohfile.WEST_R_sy,2);

  LG=quarter_number_gravitical_modes*4;
  LR=half_number_rossby_modes*2;
  nN=max_zonal_wave_number;
  nk=number_equivalent_heights;

else % netcdf
  % loading dimensions:
  nc=netcdf.open(hfile);
  [name,val]=netcdf.inqDim(nc,netcdf.inqDimID(nc,'quarter_number_gravitical_modes'));
  LG=val*4;                                                                    % number of gravity meridional modes (should be even)
  [name,val]=netcdf.inqDim(nc,netcdf.inqDimID(nc,'half_number_rossby_modes'));
  LR=val*2;                                                                    % number of Rossby meridional modes (should be even)
  [name,nN]=netcdf.inqDim(nc,netcdf.inqDimID(nc,'max_zonal_wave_number'));     % number of zonal wavenumbers
  %nk=nc.nk % may differ from number_equivalent_heights if ws0
  [name,nk]=netcdf.inqDim(nc,netcdf.inqDimID(nc,'number_equivalent_heights'));
  netcdf.close(nc);
end

nL= LG+LR;                                                                     % number of total meridional modes

fprintf(1,' - loading vertical structure functions:\n    %s\n',vfile);
if endsWith(vfile,'.mat')
  ovfile=load(vfile);
  ws0=eval(lower(ovfile.ws0));

  if ws0
    nk=nk+1;
    hk=ovfile.hk(1:nk);
    hk(1)=1.;
  else
    hk=ovfile.hk(1:nk);
  end

  Gn=ovfile.Gn(1:nk);
else
  nc=netcdf.open(vfile);
  ws0 = netcdf.getAtt(nc,netcdf.getConstant('NC_GLOBAL'),'ws0');
  ws0=eval(lower(ws0));

  hk=read_nc_var(nc,'hk');
  if ws0
    nk=nk+1;
    hk=hk(1:nk);
    hk(1)=1.;
  else
    hk=hk(1:nk);
  end

  Gn=read_nc_var(nc,'Gn');
  Gn=Gn(1:nk,:);

  netcdf.close(nc);
end
GL=size(Gn,2);

% Gaussian levels (i.e. points (Gp)) and Gaussian weights (Gw)
[Gp,Gw]=lgwt(GL,-1,1);
p_new = (Gp+1)*const.ps/2.; % Pressure levels that correspond to the chosen Gaussian sigma levels


% zonal wind, meridional wind, geopotential  -----------------------
[U_0k,U_nk]=prep(data.('u'),p_new,nk,Gn,Gw,hk,nN,ws0,'label','zonal wind');
[V_0k,V_nk]=prep(data.('v'),p_new,nk,Gn,Gw,hk,nN,ws0,'label','meridional wind');
[Z_0k,Z_nk]=prep(data.('z'),p_new,nk,Gn,Gw,hk,nN,ws0,'label','geopotential');

% Storing vertical and Fourier transforms of u, v and PHI:
W_nk = zeros([3,size(U_nk)]); % complex
W_nk(1,:,:,:,:,:)=U_nk;
W_nk(2,:,:,:,:,:)=V_nk;
W_nk(3,:,:,:,:,:)=Z_nk;

W_0k = zeros([3,size(U_0k)]);
W_0k(1,:,:,:)=U_0k;
W_0k(2,:,:,:)=V_0k;
W_0k(3,:,:,:)=Z_0k;

% Hough transforms -------------------------------------------------
disp(' - loading Hough vector functions');
if ws0, fprintf(1,'   (%s)\n',hfile); end
if endsWith(hfile,'.mat')
  HOUGH_UVZ_b   = ohfile.HOUGH_UVZ;
  HOUGH_0_UVZ_b = ohfile.HOUGH_0_UVZ;
else
  nc=netcdf.open(hfile);
  HOUGH_UVZ_b=read_nc_var(nc,'HOUGH_UVZ_real')+1j*read_nc_var(nc,'HOUGH_UVZ_imag');
  % Hough vector functions for zonal wavenumber n = 0 :
  HOUGH_0_UVZ_b=read_nc_var(nc,'HOUGH_0_UVZ_real')+1j*read_nc_var(nc,'HOUGH_0_UVZ_imag');
  netcdf.close(nc);
end

if ws0 % read barotropic file:
  fprintf(1,'   (%s)\n',hfile_B);
  if  endsWith(hfile_B,'.mat')
    ohfile_B=load(hfile_B);
    HOUGH_UVZ_B   = ohfile_B.HOUGH_UVZ;
    HOUGH_0_UVZ_B = ohfile_B.HOUGH_0_UVZ;
  else
    ncB=netcdf.open(hfile_B);
    HOUGH_UVZ_B=read_nc_var(ncB,'HOUGH_UVZ_real')+1j*read_nc_var(ncB,'HOUGH_UVZ_imag');
    % Hough vector functions for zonal wavenumber n = 0 :
    HOUGH_0_UVZ_B=read_nc_var(ncB,'HOUGH_0_UVZ');
    netcdf.close(ncB);
  end

  % concatenate:
  shape  = size(HOUGH_UVZ_b);
  shape0 = size(HOUGH_0_UVZ_b);
  shape(4)  = nk;
  shape0(3) = nk;

  HOUGH_UVZ   = zeros(shape); % complex
  HOUGH_0_UVZ = zeros(shape0); % complex

  HOUGH_UVZ(:,:,:,1,:)=HOUGH_UVZ_B;
  HOUGH_0_UVZ(:,:,1,:)=HOUGH_0_UVZ_B;

  HOUGH_UVZ(:,:,:,2:end,:)=HOUGH_UVZ_b;
  HOUGH_0_UVZ(:,:,2:end,:)=HOUGH_0_UVZ_b;
else
  HOUGH_UVZ=HOUGH_UVZ_b;
  HOUGH_0_UVZ=HOUGH_0_UVZ_b;
end


Lat=data.('u').('lat');
% check if linear or gaussian
if length(unique(diff(Lat)))==1 % linear
  Dl  = (Lat(2)-Lat(1))*pi/180; % Latitude spacing (radians)
  latType='linear';
else % gaussian
  [tmp,gw]=lgwt(length(Lat),-1,1);
  latType='gaussian'
end

nTimes=size(data.('u').('v'),1);

disp(' - computing')
THETA    = Lat*pi/180;
cosTheta = repmat(cos(THETA'),[3 1]);
w_nlk = zeros(nk,nN,nL,nTimes); % complex
for k=1:nk % vertical index
  for n=1:nN % wavenumber index
    for l=1:nL % meridional index
      for t=1:nTimes % time index
        Aux=squeeze(W_nk(:,k,:,n,t)).*squeeze(conj(HOUGH_UVZ(:,n,l,k,:))).*cosTheta; % Aux(3,Lat)
        y1=sum(Aux); % Integrand -> y1(Lat)

        if latType=='linear'
          aux1=(y1(1:end-1)+y1(2:end))*Dl/2.;
        else
          aux1=gw*y1;
        end

        w_nlk(k,n,l,t) = sum(aux1);

      end
    end
  end
end

% for zonal wavenumber n = 0:
w_0lk = zeros(nk,nL,nTimes); % complex
aux0  = zeros(length(Lat)-1); % complex
for k=1:nk % vertical index
  for l=1:nL % meridional index
    for t=1:nTimes % time index
      Aux0=squeeze(W_0k(:,k,:,t)).*squeeze(conj(HOUGH_0_UVZ(:,l,k,:))).*cosTheta; % Aux(3,Lat)
      y0=sum(Aux0); % Integrand -> y0(Lat)

      % Computes latitude integral of the Integrand using trapezoidal method
      aux0 = (y0(1:end-1)+y0(2:end))*Dl/2.;

      w_0lk(k,l,t) = sum(aux0);

    end
  end
end

if save
  fsave=save_out(struct('w_nlk',w_nlk,'w_0lk',w_0lk),varargin{:});
  varargout={w_nlk,w_0lk,fsave};
else
  varargout={w_nlk,w_0lk};
end

end


function fsave=save_out(data,varargin)
  label='out';
  format='nc'; % nc or mat
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

  attrs.platform=computer;
  attrs.environment='matlab';
  attrs.version=version;

  fsave=sprintf('%s_wnlk.%s',label,format);

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
  [nk,nN,nL,nTimes]=size(data.('w_nlk'));
  dim_nk=netcdf.defDim(nc,'number_equivalent_heights',nk);
  dim_nN=netcdf.defDim(nc,'max_zonal_wave_number',nN);
  dim_nL=netcdf.defDim(nc,'total_meridional_modes',nL); % LG+LR
  dim_nT=netcdf.defDim(nc,'time',nTimes);

  % variables:
  dim=[dim_nk,dim_nN,dim_nL,dim_nT];
  dim=dim(end:-1:1);
  var_wr=netcdf.defVar(nc,'w_nlk_real','NC_DOUBLE',dim);
  netcdf.putAtt(nc,var_wr,'long_name','Expansion coefficients (real)');
  var_wi=netcdf.defVar(nc,'w_nlk_imag','NC_DOUBLE',dim);
  netcdf.putAtt(nc,var_wi,'long_name','Expansion coefficients (imag)');

  dim=[dim_nk,dim_nL,dim_nT];
  dim=dim(end:-1:1);
  var_w0r=netcdf.defVar(nc,'w_0lk_real','NC_DOUBLE',dim);
  netcdf.putAtt(nc,var_w0r,'long_name','Zonal expansion coefficients (real)');
  var_w0i=netcdf.defVar(nc,'w_0lk_imag','NC_DOUBLE',dim);
  netcdf.putAtt(nc,var_w0i,'long_name','Zonal expansion coefficients (imag)');

  % global attributes:
  vid = netcdf.getConstant('GLOBAL');
  netcdf.putAtt(nc,vid,'date',datestr(now));
  f=fieldnames(attrs);
  for i=1:length(f)
    netcdf.putAtt(nc,vid,f{i},attrs.(f{i}));
  end

  netcdf.endDef(nc);

  % fill variables:
  netcdf.putVar(nc,var_wr,  permute(real(data.w_nlk),ndims(data.w_nlk):-1:1) );
  netcdf.putVar(nc,var_wi,  permute(imag(data.w_nlk),ndims(data.w_nlk):-1:1) );
  netcdf.putVar(nc,var_w0r, permute(real(data.w_0lk),ndims(data.w_0lk):-1:1) );
  netcdf.putVar(nc,var_w0i, permute(imag(data.w_0lk),ndims(data.w_0lk):-1:1) );
  netcdf.close(nc);
end


function [U_0k,U_nk]=prep(data,p_new,nk,Gn,Gw,hk,nN,ws0,varargin)
  constants;

  label='';

  for i=1:length(varargin)
    if isequal(varargin{i},'label')
      label=varargin{i+1};
    end
  end

  u   = data.v;
  Lat = data.lat;
  Lon = data.lon;
  P   = data.P;

  if ~ws0 & ~strcmp(label,'geopotential') % U anv V only
    u(:,end,:,:)=0; % The non-slip lower boundary condition (Tanaka and Kung, 1988)
  end

  nTimes=size(u,1);
  GL=size(Gn,2);

  u=shiftdim(u,2); % re-writes u(t,p,lat,lon) as u(lat,lon,t,p) for the 'spline' function

  fprintf(1,' - %s - interpolate p to sigma\n',label);
  us = zeros(length(Lat),length(Lon),nTimes,GL);

  for ti=1:nTimes
    us(:,:,ti,:) = spline(P,u(:,:,ti,:),p_new); % Zonal wind at the Gaussian sigma levels [1 - (-1)]
  end

  fprintf(1,' - %s - vertical transform\n',label);
  % Vertical transform
  u_k = zeros(nk,length(Lat),length(Lon),nTimes);
  for kk=1:(nk)
    Aux = 0;
    for s=1:GL
      Aux=Aux+squeeze(us(:,:,:,s)) * Gn(kk,s) * Gw(s);

      if strcmp(label,'geopotential')
        u_k(kk,:,:,:) = Aux./(const.g*hk(kk));
      else
        u_k(kk,:,:,:) = Aux./sqrt(const.g*hk(kk));
      end

    end
  end

  fprintf(1,' - %s - Fourier transform\n',label);
  % Fourier transform
  U_nk = fft(u_k,[],3);    % U_nk is the Fourier Transform of u_k along dimension 3 (i.e along longitude).
  U_nk = U_nk/length(Lon); % Scale the fft so that it is not a function of the length of input vector.

  % Retaining the first nN (see hough_functions.py) zonal wave numbers
  % U_nk has dimensions U_nk(m,Lat,nN=Lon,t)
  U_0k = U_nk(:,:,1,:);  % Wavenumber zero, i.e. the zonal mean.
  U_nk = U_nk(:,:,2:nN+1,:);   % First nN wavenumbers.

end

