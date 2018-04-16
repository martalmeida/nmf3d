function varargout=hough_functions(hk,M,nLR,nLG,latType,varargin)
%  Hough vector functions
%  The total number of the Gravity modes will be 2*nLG=nLG(east gravity)+nLG(west gravity)
%  Part I: The frequencies and the Hough functions are computed for zonal wave number m = 0
%  Part II: The frequencies and the Hough functions are computed for zonal wave numbers m > 0
%
%  M, maximum zonal wave number used in the expansion: m=0,1,...,M
%  nLR, total number of (west) Rossby modes used in the expansion (should be even)
%  nLG , half the number of Gravity modes used in the expansion (should be even)
%  latType, latitude type: linear (default, equally spaced) or gaussian
%  varargin:
%    dlat, latitude spacing if latType is linear (default is 1.5, ie, 121 points) or
%          number of gaussian lats if latType is gaussian (default is 128, corresponding
%          to a spectral truncature of T85)
%    save, create file [True]
%    format, file format: [nc] or npz
%    attrs, attributes to save [{}]
%    label, start of the saved filename ['out']
%
%  Returns baroclinic (data_b) and barotropic (data_B) data dicts and well as saved
%  filenames (fsave_b and fsvae_B) if save is True. If hk(1) is not inf (ws0 is False,
%  see vertical_structuree) no barotropic component is returned, ie:
%  -- if hk(1) is inf
%     - return data_b,data_B,fsave_b,fsave_B (if save is True)
%     - return data_b,data_B (if save is False)
%  -- if hk(1) is not inf
%     - returns data_b,fsave (if save is True)
%     - return data_b (if save is False)
%
%
%Hough vector functions as described in Swarztrauber and Kasahara (1985)
%
%References:
%  A. Kasahara (1984). The Linear Response of a Stratified Global Atmosphere to
%  Tropical Thermal Forcing, J. Atmos. Sci., 41(14). 2217--2237.
%  doi: 10.1175/1520-0469(1984)041<2217:TLROAS>2.0.CO;2
%
%  P. N. Swarztrauber and A. Kasahara (1985). The vector harmonic analysis of
%  Laplace's tidal equations, SIAM J. Sci. Stat. Comput, 6(2), 464-491.
%  doi: 10.1137/0906033
%
%  A. Kasahara (1976). Normal modes of ultralong waves in the atmosphere, Mon.
%  Weather Rev., 104(6), 669-690. doi: 10.1175/1520-0493(1976)1042.0.CO;2
%
%  Y. Shigehisa (1983). Normal Modes of the Shallow Water Equations for Zonal
%  Wavenumber Zero, J. Meteorol. Soc. Jpn., 61(4), 479-493.
%  doi: 10.2151/jmsj1965.61.4_479
%
%  A. Kasahara (1978). Further Studies on a Spectral Model of the Global
%  Barotropic Primitive Equations with Hough Harmonic Expansions, J. Atmos.
%  Sci., 35(11), 2043-2051. doi: 10.1175/1520-0469(1978)0352.0.CO;2
%

constants;

%M=42;
%nLR=40;
%nLG=20;
%latType='linear';
dlat=false;
save=true;
attrs=struct;

for i=1:length(varargin)
%  if isequal(varargin{i},'M')
%    M=varargin{i+1};
%  elseif isequal(varargin{i},'nLR')
%    nLR=varargin{i+1};
%  elseif isequal(varargin{i},'nLG')
%    nLG=varargin{i+1};
%  elseif isequal(varargin{i},'latType')
%    latType=varargin{i+1};
  if isequal(varargin{i},'dlat')
    dlat=varargin{i+1};
  elseif isequal(varargin{i},'save')
    save=varargin{i+1};
  elseif isequal(varargin{i},'attrs')
    attrs=varargin{i+1};
  end
end

if isequal(latType,'linear')
  default_dlat=1.5;
elseif isequal(latType,'gaussian')
  default_dlat=128;
end

if dlat==false
  dlat=default_dlat;
end

% keep nk as global attribute of nc files
nk=length(hk);
attrs.nk=int8(nk);
% put back attrs in varargin: (1 line of code in python!!)
varargin=cell2struct(varargin(2:2:end),varargin(1:2:end),2);
varargin.attrs=attrs;
f=fieldnames(varargin);
v=struct2cell(varargin);
varargin=cell(2*length(f),1);
varargin(1:2:end)=f;
varargin(2:2:end)=v;

% some params for saving:
params=struct;
params.M       = M;
params.nLR     = nLR;
params.nLG     = nLG;
params.NEH     = 'unk';
params.dlat    = dlat;
params.latType = latType;

if isinf(hk(1)) % ws0 True
  % baroclinic:
  [data_b,trunc,x]=hvf_baroclinic(hk(2:end),M,nLR,nLG,latType,dlat);
  params.NEH=length(hk)-1;
  if save
    fsave_b=save_out(data_b,'baroclinic',params,varargin{:});
  end

  % barotropic:
  data_B=hvf_barotropic(nLR,nLG,M,trunc,x);
  params.NEH=1;
  if save
    fsave_B=save_out(data_B,'barotropic',params,varargin{:});
  end

  if save,  varargout={data_b,data_B,fsave_b,fsave_B};
  else      varargout={data_b,data_B}
  end
else
  [data_b,trunc,x]=hvf_baroclinic(hk,M,nLR,nLG,latType,dlat);
  params.NEH=length(hk);
  if save
    fsave=save_out(data_b,'ws0False',params,varargin{:});
    varargout={data_b,fsave};
  else varargout={data_b};
  end
end
end


function fsave=save_out(data,tag,params,varargin)
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

attrs.platform=computer;
attrs.environment='matlab';
attrs.version=version;

fsave=sprintf('%s_hvf_M%d_nLR%d_nLG%d_NEH%d_dlat%s%s_%s.%s',label,params.M,params.nLR,params.nLG*2,...
                                           params.NEH,num2str(params.dlat),params.latType,tag,format);
fprintf(1,'saving %s\n',fsave);
if isequal(format,'mat')
  % update data with attrs:
  f = fieldnames(attrs);
  for i = 1:length(f)
    data.(f{i})=attrs.(f{i});
  end
  save(fsave, '-struct','data');
elseif isequal(format,'nc')
  if isequal(tag,'barotropic')
    save_nc_bar(fsave,data,attrs);
  else
    save_nc(fsave,data,attrs);
  end
else
  disp('Unknown format, use nc or mat');
end

end


function save_nc(fname,data,attrs)
nc=netcdf.create(fname,'CLOBBER');

% dimensions:
[some,M,L,NEH,lat]=size(data.HOUGH_UVZ);
quarter_nLG=size(data.WEST_G_sy,2);
half_nLR=size(data.WEST_R_sy,2);
Np1=size(data.WEST_R_0_sy,1);

dim_3   = netcdf.defDim(nc,'components_uvz',3);              % components uvz
dim_M   = netcdf.defDim(nc,'max_zonal_wave_number',M);       % max zonal wave number (M)
dim_L   = netcdf.defDim(nc,'number_meridional_modes',L);     % total number meridional modes (L, must be even!)
dim_lat = netcdf.defDim(nc,'lat',lat);                       % n lats
dim_NEH = netcdf.defDim(nc,'number_equivalent_heights',NEH); % number equivalent heights (NEH)

dim_qg  = netcdf.defDim(nc,'quarter_number_gravitical_modes',quarter_nLG);
dim_hr  = netcdf.defDim(nc,'half_number_rossby_modes',half_nLR);
dim_Np1 = netcdf.defDim(nc,'Np1',Np1);

% Note that L=NG+NR=4*quarter_number_gravitical_modes+2*half_number_rossby_modes)

% variables:
%   hough:
k='HOUGH_UVZ';
dim=[dim_3,dim_M,dim_L,dim_NEH,dim_lat];
dim=dim(end:-1:1);
var_hr=netcdf.defVar(nc,[k '_real'],'NC_DOUBLE',dim);
netcdf.putAtt(nc,var_hr,'long_name','hough functions - eddies (real)');
var_hi=netcdf.defVar(nc,[k '_imag'],'NC_DOUBLE',dim);
netcdf.putAtt(nc,var_hi,'long_name','hough functions - eddies (imag)');

k='HOUGH_0_UVZ';
dim=[dim_3,dim_L,dim_NEH,dim_lat];
dim=dim(end:-1:1);
var_h0r=netcdf.defVar(nc,[k '_real'],'NC_DOUBLE',dim);
netcdf.putAtt(nc,var_h0r,'long_name','hough functions - zonal mean (real)');
var_h0i=netcdf.defVar(nc,[k '_imag'],'NC_DOUBLE',dim);
netcdf.putAtt(nc,var_h0i,'long_name','hough functions - zonal mean (imag)');

%   westward - eddies:
dim=[dim_M,dim_qg,dim_NEH];
dim=dim(end:-1:1);

k='WEST_G_sy';
var_wgs=netcdf.defVar(nc,k,'NC_DOUBLE',dim);
netcdf.putAtt(nc,var_wgs,'long_name','frequencies of the symmetric westward gravity waves - eddies');

k='WEST_G_asy';
var_wga=netcdf.defVar(nc,k,'NC_DOUBLE',dim);
netcdf.putAtt(nc,var_wga,'long_name','frequencies of the antisymmetric westward gravity waves - eddies');

dim=[dim_M,dim_hr,dim_NEH];
dim=dim(end:-1:1);

k='WEST_R_sy';
var_wrs=netcdf.defVar(nc,k,'NC_DOUBLE',dim);
netcdf.putAtt(nc,var_wrs,'long_name','frequencies of the symmetric westward Rossby waves - eddies');

k='WEST_R_asy';
var_wra=netcdf.defVar(nc,k,'NC_DOUBLE',dim);
netcdf.putAtt(nc,var_wra,'long_name','frequencies of the antisymmetric westward Rossby waves - eddies');

%   eastward - eddies:
dim=[dim_M,dim_qg,dim_NEH];
dim=dim(end:-1:1);

k='EAST_G_sy';
var_egs=netcdf.defVar(nc,k,'NC_DOUBLE',dim);
netcdf.putAtt(nc,var_egs,'long_name','frequencies of the symmetric eastward gravity waves - eddies');

k='EAST_G_asy';
var_ega=netcdf.defVar(nc,k,'NC_DOUBLE',dim);
netcdf.putAtt(nc,var_ega,'long_name','frequencies of the antisymmetric eastward gravity waves - eddies');

%   westward - zonal mean:
dim=[dim_qg,dim_NEH];
dim=dim(end:-1:1);

k='WEST_G_0_sy';
var_wg0s=netcdf.defVar(nc,k,'NC_DOUBLE',dim);
netcdf.putAtt(nc,var_wg0s,'long_name','frequencies of the symmetric westward gravity waves - zonal mean');

k='WEST_G_0_asy';
var_wg0a=netcdf.defVar(nc,k,'NC_DOUBLE',dim);
netcdf.putAtt(nc,var_wg0a,'long_name','frequencies of the antisymmetric westward gravity waves - zonal mean');

dim=[dim_hr,dim_NEH];
dim=dim(end:-1:1);

k='WEST_R_0_sy';
var_wr0s=netcdf.defVar(nc,k,'NC_DOUBLE',dim);
netcdf.putAtt(nc,var_wr0s,'long_name','frequencies of the symmetric westward Rossby waves - zonal mean');

k='WEST_R_0_asy';
var_wr0a=netcdf.defVar(nc,k,'NC_DOUBLE',dim);
netcdf.putAtt(nc,var_wr0a,'long_name','frequencies of the antisymmetric westward Rossby waves - zonal mean');

%   eastward - zonal mean:
dim=[dim_qg,dim_NEH];
dim=dim(end:-1:1);

k='EAST_G_0_sy';
var_eg0s=netcdf.defVar(nc,k,'NC_DOUBLE',dim);
netcdf.putAtt(nc,var_eg0s,'long_name','frequencies of the symmetric eastward gravity waves - zonal mean');

k='EAST_G_0_asy';
var_eg0a=netcdf.defVar(nc,k,'NC_DOUBLE',dim);
netcdf.putAtt(nc,var_eg0a,'long_name','frequencies of the antisymmetric eastward gravity waves - zonal mean');

% global attributes:
vid = netcdf.getConstant('GLOBAL');
netcdf.putAtt(nc,vid,'date',datestr(now));
f=fieldnames(attrs);
for i=1:length(f)
  netcdf.putAtt(nc,vid,f{i},attrs.(f{i}));
end

netcdf.endDef(nc);

% fill variables:
netcdf.putVar(nc,var_hr,   permute(real(data.HOUGH_UVZ),   ndims(data.HOUGH_UVZ)    :-1:1) );
netcdf.putVar(nc,var_hi,   permute(imag(data.HOUGH_UVZ),   ndims(data.HOUGH_UVZ)    :-1:1) );
netcdf.putVar(nc,var_h0r,  permute(real(data.HOUGH_0_UVZ), ndims(data.HOUGH_0_UVZ)  :-1:1) );
netcdf.putVar(nc,var_h0i,  permute(imag(data.HOUGH_0_UVZ), ndims(data.HOUGH_0_UVZ)  :-1:1) );
netcdf.putVar(nc,var_wgs,  permute(data.WEST_G_sy,         ndims(data.WEST_G_sy)    :-1:1) );
netcdf.putVar(nc,var_wga,  permute(data.WEST_G_asy,        ndims(data.WEST_G_asy)   :-1:1) );
netcdf.putVar(nc,var_wrs,  permute(data.WEST_R_sy,         ndims(data.WEST_R_sy)    :-1:1) );
netcdf.putVar(nc,var_wra,  permute(data.WEST_R_asy,        ndims(data.WEST_R_asy)   :-1:1) );
netcdf.putVar(nc,var_egs,  permute(data.EAST_G_sy,         ndims(data.EAST_G_sy)    :-1:1) );
netcdf.putVar(nc,var_ega,  permute(data.EAST_G_asy,        ndims(data.EAST_G_asy)   :-1:1) );
netcdf.putVar(nc,var_wg0s, permute(data.WEST_G_0_sy,       ndims(data.WEST_G_0_sy)  :-1:1) );
netcdf.putVar(nc,var_wg0a, permute(data.WEST_G_0_asy,      ndims(data.WEST_G_0_asy) :-1:1) );
netcdf.putVar(nc,var_wr0s, permute(data.WEST_R_0_sy,       ndims(data.WEST_R_0_sy)  :-1:1) );
netcdf.putVar(nc,var_wr0a, permute(data.WEST_R_0_asy,      ndims(data.WEST_R_0_asy) :-1:1) );
netcdf.putVar(nc,var_eg0s, permute(data.EAST_G_0_sy,       ndims(data.EAST_G_0_sy)  :-1:1) );
netcdf.putVar(nc,var_eg0a, permute(data.EAST_G_0_asy,      ndims(data.EAST_G_0_asy) :-1:1) );

netcdf.close(nc);
end


function save_nc_bar(fname,data,attrs)
nc=netcdf.create(fname,'CLOBBER');

% dimensions:
[some,M,L,lat]=size(data.HOUGH_UVZ);
nLR=size(data.SIGMAS,2);

dim_3   = netcdf.defDim(nc,'components_uvz',3);          % components uvz
dim_M   = netcdf.defDim(nc,'max_zonal_wave_number',M);   % max zonal wave number (M)
dim_L   = netcdf.defDim(nc,'number_meridional_modes',L); % total number meridional modes (L, must be even!)
dim_lat = netcdf.defDim(nc,'lat',lat);                   % n lats
dim_r   = netcdf.defDim(nc,'number_Rossby_modes',nLR);   % total number of (west) Rossby modes (must be even!)

% variables:
%   hough and hough to reconstr.:
k='HOUGH_UVZ';
dim=[dim_3,dim_M,dim_L,dim_lat];
dim=dim(end:-1:1);
var_hr=netcdf.defVar(nc,[k '_real'],'NC_DOUBLE',dim);
netcdf.putAtt(nc,var_hr,'long_name','hough functions - eddies (real)')
var_hi=netcdf.defVar(nc,[k '_imag'],'NC_DOUBLE',dim);
netcdf.putAtt(nc,var_hi,'long_name','hough functions - eddies (imag)')

k='HOUGH_UVZ_2rec';
var_h2rr=netcdf.defVar(nc,[k '_real'],'NC_DOUBLE',dim);
netcdf.putAtt(nc,var_h2rr,'long_name','hough functions for reconstruction - eddies (real)');
var_h2ri=netcdf.defVar(nc,[k '_imag'],'NC_DOUBLE',dim);
netcdf.putAtt(nc,var_h2ri,'long_name','hough functions for reconstruction - eddies (imag)');

k='HOUGH_0_UVZ';
dim=[dim_3,dim_L,dim_lat];
dim=dim(end:-1:1);
var_h0=netcdf.defVar(nc,k,'NC_DOUBLE',dim);
netcdf.putAtt(nc,var_h0,'long_name','hough functions - zonal mean');

k='HOUGH_0_UVZ_2rec';
var_h02r=netcdf.defVar(nc,k,'NC_DOUBLE',dim);
netcdf.putAtt(nc,var_h02r,'long_name','hough functions for reconstruction - zonal mean');

%   sigmas
k='SIGMAS';
dim=[dim_M,dim_r];
dim=dim(end:-1:1);
var_s=netcdf.defVar(nc,k,'NC_DOUBLE',dim);
netcdf.putAtt(nc,var_s,'long_name','Haurwitz frequencies');

% global attributes:
vid = netcdf.getConstant('GLOBAL');
netcdf.putAtt(nc,vid,'date',datestr(now));
f=fieldnames(attrs);
for i=1:length(f)
  netcdf.putAtt(nc,vid,f{i},attrs.(f{i}));
end

netcdf.endDef(nc);

% fill variables:
netcdf.putVar(nc,var_hr,   permute(real(data.HOUGH_UVZ),      ndims(data.HOUGH_UVZ)        :-1:1) );
netcdf.putVar(nc,var_hi,   permute(imag(data.HOUGH_UVZ),      ndims(data.HOUGH_UVZ)        :-1:1) );
netcdf.putVar(nc,var_h2rr, permute(real(data.HOUGH_UVZ_2rec), ndims(data.HOUGH_UVZ_2rec)   :-1:1) );
netcdf.putVar(nc,var_h2ri, permute(imag(data.HOUGH_UVZ_2rec), ndims(data.HOUGH_UVZ_2rec)   :-1:1) );
netcdf.putVar(nc,var_h0,   permute(data.HOUGH_0_UVZ,          ndims(data.HOUGH_0_UVZ)      :-1:1) );
netcdf.putVar(nc,var_h02r, permute(data.HOUGH_0_UVZ_2rec,     ndims(data.HOUGH_0_UVZ_2rec) :-1:1) );
netcdf.putVar(nc,var_s,    permute(data.SIGMAS,               ndims(data.SIGMAS)           :-1:1) );

netcdf.close(nc);
end
