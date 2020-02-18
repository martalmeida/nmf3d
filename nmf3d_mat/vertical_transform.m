function varargout=vertical_transform(u,hk,nk,Gn,p_old,p_new,dataLabel,varargin)
%  Vertical transform
%
%  u, variable defined at pressure levels
%  hk, equivalent heights
%  nk, total number of equivalent heights
%  Gn, vertical structure functions
%  p_old, original pressure levels
%  p_new, Gaussian pressure levels
%  dataLabel, variable type:
%    'zonal wind', u-wind
%    'meridional wind', v-wind
%    'geopotential', phi (z)
%    'I1', zonal component of the kinetic energy interaction term (square brakets of eq. A16, ref1)
%    'I2', meridional component of the kinetic energy interaction term (square brakets of eq. A17, ref1)
%    'J3', available potential energy interaction term (square brakets of eq. A18, ref1)
%  varargin:
%    save, create file [true]
%    format, file format: [nc] or mat
%    attrs, attributes to save [{}]
%    label, start of the saved filename ['out']
%
%   Returns the vertical transform as well as saved filename if save is true
%
%  References
%    ref1:
%    Castanheira, JM, Marques, CAF (2019). The energy cascade associated with
%    daily variability of the North Atlantic Oscillation, Q J R Meteorol Soc.,
%    145: 197– 210. doi: https://doi.org/10.1002/qj.3422
%

constants;

save=false; % create file
for i=1:length(varargin)
  if isequal(varargin{i},'save')
    save=varargin{i+1};
  end
end

ws0=isinf(hk(1));
hk=hk(1:nk);
if ws0
  hk(1)=1.;
end

GL=size(Gn,2);
[nTimes,nz,nLat,nLon]=size(u);

[Gp,Gw]=lgwt(GL,-1,1);

if ~ws0 & (strcmp(dataLabel,'zonal wind') | strcmp(dataLabel,'meridional wind'))% U and V only
  u(:,end,:,:)=0; % The non-slip lower boundary condition (Tanaka and Kung, 1988)
end

u=shiftdim(u,2); % re-writes u(t,p,lat,lon) as u(lat,lon,t,p) for the 'spline' function

fprintf(1,' - %s - interpolate p to sigma\n',dataLabel);
us = zeros(nLat,nLon,nTimes,GL);

for ti=1:nTimes
  us(:,:,ti,:) = spline(p_old,u(:,:,ti,:),p_new); % variable at the Gaussian sigma levels [1 - (-1)]
end

% vertical derivative of Gn
if strcmp(dataLabel,'J3')
  Dp=diff(p_new);
  d_Gn_dpnew=zeros(size(Gn));
  if isinf(hk(1)) % ws0 True
    d_Gn_dpnew(:,1)=0; % lower boundary condition
  else
    d_Gn_dpnew(:,1)=(Gn(:,2)-Gn(:,1))/Dp(1); % forward differences
  end
  d_Gn_dpnew(:,GL)=(Gn(:,end)-Gn(:,end-1))/Dp(end); % backward differences
  for p=2:GL-1
    d_Gn_dpnew(:,p)=(Dp(p-1)^2*Gn(:,p+1)-Dp(p)^2*Gn(:,p-1)-(Dp(p-1)^2-Dp(p)^2)*Gn(:,p))/(Dp(p-1)*Dp(p)*(Dp(p-1)+Dp(p)));
  end
end

fprintf(1,' - %s - vertical transform\n',dataLabel);
% Vertical transform
u_k = zeros(nk,nLat,nLon,nTimes);
for kk=1:nk
  Aux = 0;
  for s=1:GL
    if strcmp(dataLabel,'J3')
      Aux=Aux-0.5*squeeze(us(:,:,:,s))*d_Gn_dpnew(kk,s)*Gw(s);
    else
      Aux=Aux+0.5*squeeze(us(:,:,:,s))*Gn(kk,s)*Gw(s);
    end

    if strcmp(dataLabel,'geopotential')
      u_k(kk,:,:,:) = Aux./(const.g*hk(kk));
    elseif  strcmp(dataLabel,'zonal wind') | strcmp(dataLabel,'meridional wind')
      u_k(kk,:,:,:) = Aux./sqrt(const.g*hk(kk));
    elseif  strcmp(dataLabel,'I1') | strcmp(dataLabel,'I2')
      u_k(kk,:,:,:) = Aux./(2*const.Om*sqrt(const.g*hk(kk)));
    elseif  strcmp(dataLabel,'J3')
      u_k(kk,:,:,:) = Aux./(2*const.Om);
    end

  end
end

if save
  if     strcmp(dataLabel,'geopotential'),    vname='z_k';
  elseif strcmp(dataLabel,'zonal wind'),      vname='u_k';
  elseif strcmp(dataLabel,'meridional wind'), vname='v_k';
  elseif strcmp(dataLabel,'I1'),              vname='I1_k';
  elseif strcmp(dataLabel,'I2'),              vname='I2_k';
  elseif strcmp(dataLabel,'J3'),              vname='J3_k';
  end

  fsave=save_out(struct(vname,u_k),varargin{:});
  varargout={u_k,fsave};
else
  varargout={u_k};
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

  if     isfield(data,'z_k'),  label2='z';
  elseif isfield(data,'u_k'),  label2='u';
  elseif isfield(data,'v_k'),  label2='v';
  elseif isfield(data,'I1_k'), label2='I1';
  elseif isfield(data,'I2_k'), label2='I2';
  elseif isfield(data,'J3_k'), label2='J3';
  end

  fsave=sprintf('%s_%s_k.%s',label,label2,format);

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

  if isfield(data,'z_k')
    vname='z_k';
    vlname='Vertical transform of geopotential';
  elseif isfield(data,'u_k')
    vname='u_k';
    vlname='Vertical transform of zonal wind';
  elseif isfield(data,'v_k')
    vname='v_k';
    vlname='Vertical transform of meridonal wind';
  elseif isfield(data,'I1_k')
    vname='I1_k';
    vlname='Vertical transform of term I1';
  elseif isfield(data,'I2_k')
    vname='I2_k';
    vlname='Vertical transform of term I2';
  elseif isfield(data,'J3_k')
    vname='J3_k';
    vlname='Vertical transform of term J3';
  end
  v=data.(vname);

  % dimensions:
  [nk,nLat,nLon,nTime]=size(v);
  dim_nk   = netcdf.defDim(nc,'number_equivalent_heights',nk);
  dim_nlat = netcdf.defDim(nc,'latitude',nLat);
  dim_nlon = netcdf.defDim(nc,'longitude',nLon);
  dim_nt   = netcdf.defDim(nc,'time',nTime);

  % variables:
  dim=[dim_nk,dim_nlat,dim_nlon,dim_nt];
  dim=dim(end:-1:1);
  var=netcdf.defVar(nc,vname,'NC_DOUBLE',dim);
  netcdf.putAtt(nc,var,'long_name',vlname);

  % global attributes:
  vid = netcdf.getConstant('GLOBAL');
  netcdf.putAtt(nc,vid,'date',datestr(now));
  f=fieldnames(attrs);
  for i=1:length(f)
    netcdf.putAtt(nc,vid,f{i},attrs.(f{i}));
  end

  netcdf.endDef(nc);

  % fill variables:
  netcdf.putVar(nc,var,  permute(real(v),ndims(v):-1:1) );
  netcdf.close(nc);
end
