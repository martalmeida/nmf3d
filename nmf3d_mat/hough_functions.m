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
%    format, file format: [nc] or mat
%    attrs, attributes to save [{}]
%    label, start of the saved filename ['out']
%
%  Returns concatenated baroclinic and barotropic data as well as saved filename
%  if save is True. If hk(1) is not inf (ws0 is False, see vertical_structure)
%  no barotropic component is returned
%
%  Hough vector functions as described in Swarztrauber and Kasahara (1985)
%
%  References:
%    A. Kasahara (1976). Normal modes of ultralong waves in the atmosphere, Mon.
%    Weather Rev., 104(6), 669-690. doi: 10.1175/1520-0493(1976)1042.0.CO;2
%
%    A. Kasahara (1978). Further Studies on a Spectral Model of the Global
%    Barotropic Primitive Equations with Hough Harmonic Expansions, J. Atmos.
%    Sci., 35(11), 2043-2051. doi: 10.1175/1520-0469(1978)0352.0.CO;2
%
%    Y. Shigehisa (1983). Normal Modes of the Shallow Water Equations for Zonal
%    Wavenumber Zero, J. Meteorol. Soc. Jpn., 61(4), 479-493.
%    doi: 10.2151/jmsj1965.61.4_479
%
%    A. Kasahara (1984). The Linear Response of a Stratified Global Atmosphere to
%    Tropical Thermal Forcing, J. Atmos. Sci., 41(14). 2217--2237.
%    doi: 10.1175/1520-0469(1984)041<2217:TLROAS>2.0.CO;2
%
%    P. N. Swarztrauber and A. Kasahara (1985). The vector harmonic analysis of
%    Laplace's tidal equations, SIAM J. Sci. Stat. Comput, 6(2), 464-491.
%    doi: 10.1137/0906033

constants;

dlat=false;
save=true;
attrs=struct;

for i=1:length(varargin)
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

% some params for saving:
params=struct;
params.M       = M;
params.nLR     = nLR;
params.nLG     = nLG;
params.NEH     = length(hk);
params.dlat    = dlat;
params.latType = latType;

L=nLR + 2*nLG;

if isinf(hk(1)) % ws0 True
  % baroclinic:
  [data_b,trunc,x]=hvf_baroclinic(hk(2:end),M,nLR,nLG,latType,dlat);

  % barotropic:
  data_B=hvf_barotropic(nLR,nLG,M,trunc,x);

  % concatenate barotropic and baroclinic, Hough functions:
  % zonal:
  HOUGHs_0_UVZ=zeros(3,L,length(hk),length(x));
  HOUGHs_0_UVZ(:,:,1,:)     = data_B.HOUGH_0_UVZ;
  HOUGHs_0_UVZ(:,:,2:end,:) = data_b.HOUGH_0_UVZ;

  % eddies:
  HOUGHs_m_UVZ=zeros(3,M,L,length(hk),length(x));
  HOUGHs_m_UVZ(:,:,:,1,:)     = data_B.HOUGH_UVZ;
  HOUGHs_m_UVZ(:,:,:,2:end,:) = data_b.HOUGH_UVZ;

  % concatenate zonal and eddies:
  HOUGHs_UVZ=zeros(3,M+1,L,length(hk),length(x));
  HOUGHs_UVZ(:,1,:,:,:)     = HOUGHs_0_UVZ;
  HOUGHs_UVZ(:,2:end,:,:,:) = HOUGHs_m_UVZ;


  % concatenate barotropic and baroclinic, frequencies:
  % zonal:
  FREQs_0=zeros(L,length(hk));
  FREQs_0(:,2:end) = data_b.FREQS_0;

  % eddies:
  sigmas=zeros(M,L)*nan;
  sigmas(:,end-nLR+1:end)=data_B.SIGMAS;
  FREQs_m= zeros(M,L,length(hk));
  FREQs_m(:,:,1)     = sigmas;
  FREQs_m(:,:,2:end) = data_b.FREQS_m;

  % concatenate zonal and eddies:
  FREQs= zeros(M+1,L,length(hk));
  FREQs(1,:,:)     = FREQs_0;
  FREQs(2:end,:,:) = FREQs_m;

  % data to store:
  data=struct;
  data.HOUGHs_UVZ = HOUGHs_UVZ;
  data.FREQs      = FREQs;

  if save
    fsave=save_out(data,'ws0True',params,varargin{:});
    varargout={data,fsave};
  else
    varargout={data};
  end

else
  [data_b,trunc,x]=hvf_baroclinic(hk,M,nLR,nLG,latType,dlat);

  % concatenate zonal and eddies, Hough functions and frequencies:
  HOUGHs_UVZ=zeros(3,M+1,L,length(hk),length(x));
  HOUGHs_UVZ(:,1,:,:,:)     = data_b.HOUGH_0_UVZ;
  HOUGHs_UVZ(:,2:end,:,:,:) = data_b.HOUGH_UVZ;

  FREQs=zeros(M+1,L,length(hk));
  FREQs(1,:,:)     = data_b.FREQS_0;
  FREQs(2:end,:,:) = data_b.FREQS_m;

  % data to store:
  data=struct;
  data.HOUGHs_UVZ = HOUGHs_UVZ;
  data.FREQs      = FREQs;

  if save
    fsave=save_out(data,'ws0False',params,varargin{:});
    varargout={data,fsave};
  else
    varargout={data};
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
    save_nc(fsave,data,attrs);
  else
    disp('Unknown format, use nc or mat');
  end
end


function save_nc(fname,data,attrs)
  nc=netcdf.create(fname,'CLOBBER');

  % dimensions:
  [some,M_,L,NEH,lat]=size(data.HOUGHs_UVZ);
  dim_3   = netcdf.defDim(nc,'components_uvz',3);              % components uvz
  dim_M   = netcdf.defDim(nc,'max_zonal_wave_number_and_zonal_mean',M_); % max zonal wave number (M)
  dim_L   = netcdf.defDim(nc,'number_meridional_modes',L);     % total number meridional modes (L, must be even!)
  dim_lat = netcdf.defDim(nc,'lat',lat);                       % n lats
  dim_NEH = netcdf.defDim(nc,'number_equivalent_heights',NEH); % number equivalent heights (NEH)

  % variables:
  %   hough:
  k='HOUGHs_UVZ';
  dim=[dim_3,dim_M,dim_L,dim_NEH,dim_lat];
  dim=dim(end:-1:1);
  var_hr=netcdf.defVar(nc,[k '_real'],'NC_DOUBLE',dim);
  netcdf.putAtt(nc,var_hr,'long_name','hough functions - real');
  var_hi=netcdf.defVar(nc,[k '_imag'],'NC_DOUBLE',dim);
  netcdf.putAtt(nc,var_hi,'long_name','hough functions - imag');

  % frequencies:
  k='FREQs';
  dim=[dim_M,dim_L,dim_NEH];
  dim=dim(end:-1:1);
  var_f=netcdf.defVar(nc,k,'NC_DOUBLE',dim);
  netcdf.putAtt(nc,var_f,'long_name','frequencies');

  % global attributes:
  vid = netcdf.getConstant('GLOBAL');
  netcdf.putAtt(nc,vid,'date',datestr(now));
  f=fieldnames(attrs);
  for i=1:length(f)
    netcdf.putAtt(nc,vid,f{i},attrs.(f{i}));
  end

  netcdf.endDef(nc);

  % fill variables:
  netcdf.putVar(nc,var_hr, permute(real(data.HOUGHs_UVZ),  ndims(data.HOUGHs_UVZ) :-1:1) );
  netcdf.putVar(nc,var_hi, permute(imag(data.HOUGHs_UVZ),  ndims(data.HOUGHs_UVZ) :-1:1) );
  netcdf.putVar(nc,var_f,  permute(data.FREQs,             ndims(data.FREQs)      :-1:1) );
  netcdf.close(nc);
end
