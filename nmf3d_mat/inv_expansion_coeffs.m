function varargout=inv_expansion_coeffs(vfile,hfile,wfile,zi,mi,vi,pl,lon,varargin)
%  Inverse expansion coefficients
%  Compute the inverse of Vertical, Fourier and Hough transforms.
%  Recovers the zonal and meridional wind, and geopotential perturbation
%  (from the reference geopotential) in the physical space, for the chosen
%  set of modes. Uses equivalent heights, vertical structure functions
%  Hough functions and complex expansion coefficients.
%
%  vfile: file with the equivalent heights (hk) and vertical structure functions (Gn)
%  hfile: file with the Hough functions (HOUGHs_UVZ)
%  wfile: file with the expansion coefficients (w_nlk)
%  zi:    wavenumber indices, zi=0 (Zonal mean) and zi>0 (Eddies)
%  mi:    meridional indices
%  vi:    vertical indices, vi=0 (Barotropic mode) and vi>0 (Baroclinic modes)
%  pl:    pressure levels (hPa units)
%  lon:   longitudes (deg units)
%  varargin:
%    uvz:   components of wind and mass fields to calculate
%           - default [1 1 1] - calculate the zonal and meridional wind and the geopotential (u, v, z)
%    save, create file [true]
%    format, file format: [nc] or mat
%    attrs, attributes to save [{}]
%    label, filename to save, without extension ['outinv_uvz']
%
%  Returns the reconstructed  zonal and meridional wind, and geopotential perturbation,
%  as well as saved filename if save is true

constants;

uvz=[1,1,1];
save=true; % create file
for i=1:length(varargin)
  if isequal(varargin{i},'uvz')
    uvz=varargin{i+1};
  elseif isequal(varargin{i},'save')
    save=varargin{i+1};
  end
end

% The wavenumber, meridional and vertical indices
ZI = zi+1;
MI = mi;
VI = vi+1;

lon=lon*pi/180; % lon in radians

fprintf(1,' - loading vertical structure functions:\n    %s\n',vfile);
if ends_with(vfile,'.mat')
  ovfile=load(vfile);
  ws0=eval(lower(ovfile.ws0));

  hk    = ovfile.hk;
  Gn    = ovfile.Gn;
  p_new = ovfile.plev;
else
  nc=netcdf.open(vfile);
  ws0 = netcdf.getAtt(nc,netcdf.getConstant('NC_GLOBAL'),'ws0');
  ws0 = eval(lower(ws0));

  hk    = read_nc_var(nc,'hk');
  Gn    = read_nc_var(nc,'Gn');
  p_new = read_nc_var(nc,'plev');

  netcdf.close(nc);
end

fprintf(1,' - loading Hough vector functions:\n    %s\n',hfile);
if ends_with(hfile,'.mat')
  ohfile=load(hfile);
  HOUGHs_UVZ = ohfile.HOUGHs_UVZ;
  lat = ohfile.lat;
else
  nc=netcdf.open(hfile);
  HOUGHs_UVZ=read_nc_var(nc,'HOUGHs_UVZ_real')+1j*read_nc_var(nc,'HOUGHs_UVZ_imag');
  lat=read_nc_var(nc,'lat');
  netcdf.close(nc);
end

fprintf(1,' - loading expansion coefficients:\n    %s\n',wfile);
if ends_with(wfile,'.mat')
  owfile=load(wfile);
  w_nlk = owfile.w_nlk;
else
  nc=netcdf.open(wfile);
  w_nlk=read_nc_var(nc,'w_nlk_real')+1j*read_nc_var(nc,'w_nlk_imag');
  netcdf.close(nc);
end

[three,nN,nmm,nk,nLat]=size(HOUGHs_UVZ);
hk=hk(1:nk);
nT=size(w_nlk,ndims(w_nlk));

if ws0
    hk(1) = 1;
end

% Interpolates vertical structure functions (Gn) for level p with cubic spline
plev=p_new;
plev(1)=const.ps; % surface max pressure, to avoid extrap!
for i=1:nk
  Gn_interpS(i,:) = spline(plev,Gn(i,:),pl*100); % plev pl*100 in Pa
end

% Permuting dimensions of Fourier-Hough transforms for convenience
w_nlk = permute(w_nlk,[2 3 1 4]); % size(w_nlk) = (n,l,k,t)

if max(ZI)>nN
  error('maximum wavenumber was exceeded')
end

if max(VI)>nk
  error('maximum vertical index was exceeded')
end

% variables for the horizontal wind and mass fields expansion into 3-D normal modes
un = zeros(nN, nLat, numel(lon)); % size(u) = (n,lat,lon)
vn = un;
zn = un;

u = zeros(nT, numel(pl), nLat, numel(lon)); % size(u) = (nT,npL,lat,lon)
v = u;
z = u;

select_nlk = zeros(nN,nmm,nk);
select_nlk(ZI,MI,VI) = 1;

fprintf(1,' - computing\n');
for ip=1:numel(pl) % loop over p levels
  Gn = Gn_interpS(:,ip);
  for it=1:nT % loop over times
    A_w_nlk = select_nlk.*squeeze(w_nlk(:,:,:,it));

    % Replicating variables into arrays with matching sizes for
    % elementwise array multiplications (element-by-element). The sizes
    % must mach for dimensions (n,l,k)
    A_h_k = permute(repmat(hk,[1 nN nmm]),[2 3 1]);          % size(A_h_k) = (n,l,k)
    A_Gn = permute(squeeze(repmat(Gn,[1 nN  nmm])),[2 3 1]); % size(A_Gn)  = (n,l,k)

    nW = (0:nN-1).';
    A_Fourier   = repmat(exp(const.I*nW*lon),[1 1 nmm nk]);  % size(A_Fourier)   = (n,lon,l,k)

    for la=1:nLat
      for lo=1:numel(lon)
        % summing over chosen vertical and meridional modes

        un(:,la,lo) = (2*real(squeeze(sum(sum(A_w_nlk .* sqrt(const.g.*A_h_k) .* A_Gn .* squeeze(HOUGHs_UVZ(1,:,:,:,la)) .* squeeze(A_Fourier(:,lo,:,:)),3),2)))).';
        vn(:,la,lo) = (2*real(squeeze(sum(sum(A_w_nlk .* sqrt(const.g.*A_h_k) .* A_Gn .* squeeze(HOUGHs_UVZ(2,:,:,:,la)) .* squeeze(A_Fourier(:,lo,:,:)),3),2)))).';
        zn(:,la,lo) = (2*real(squeeze(sum(sum(A_w_nlk .*      const.g.*A_h_k  .* A_Gn .* squeeze(HOUGHs_UVZ(3,:,:,:,la)) .* squeeze(A_Fourier(:,lo,:,:)),3),2)))).';
      end
    end

    u(it,ip,:,:) = squeeze( sum(un(2:end,:,:),1) + 0.5*un(1,:,:) );
    v(it,ip,:,:) = squeeze( sum(vn(2:end,:,:),1) + 0.5*vn(1,:,:) );
    z(it,ip,:,:) = squeeze( sum(zn(2:end,:,:),1) + 0.5*zn(1,:,:) );
  end
end


data=struct;
data.lon = lon*180/pi;
data.lat = lat;
if uvz(1), data.u=u; end
if uvz(2), data.v=v; end
if uvz(3), data.z=z; end

if save
  % add some input data as attributes:
  attrs=struct;
  j=0;
  for i=1:length(varargin)
    if isequal(varargin{i},'attrs')
      attrs=varargin{i+1};
      j=i;
    end
  end
  attrs.zi  = int32(zi);
  attrs.mi  = int32(mi);
  attrs.vi  = int32(vi);
  attrs.pl  = pl;
  attrs.uvz = int32(uvz);
  if j==0
    varargin{end+1}='attrs';
    varargin{end+1}=attrs;
  else
    varargin{j}=attrs;
  end

  fsave=save_out(data,varargin{:});
  varargout={data,fsave};
else
  varargout={data};
end

end


function tf = ends_with(str, suffix)
  % Return true if the string ends in the specified suffix
  % Matlab >2016b already includes the function endsWith
  n = length(suffix);
  if length(str) < n
    tf =  false;
  else
    tf = strcmp(str(end-n+1:end), suffix);
  end
end


function fsave=save_out(data,varargin)
  label='outinv_uvz';
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

  fsave=sprintf('%s.%s',label,format);

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
  addu=isfield(data,'u');
  addv=isfield(data,'v');
  addz=isfield(data,'z');

  nc=netcdf.create(fname,'CLOBBER');

  % dimensions:
  if     addu, tmp=data.u;
  elseif addv, tmp=data.v;
  elseif addz, tmp=data.z;
  end

  if addu | addv | addz
    [nT,nLev,nLat,nLon]=size(tmp);

    dim_nT  = netcdf.defDim(nc,'time',nT);
    dim_nP  = netcdf.defDim(nc,'pressure_levels',nLev);
    dim_lat = netcdf.defDim(nc,'lat',nLat);
    dim_lon = netcdf.defDim(nc,'lon',nLon);

    dim=[dim_nT,dim_nP,dim_lat,dim_lon];
    dim=dim(end:-1:1);
  else
    dim_lat = netcdf.defDim(nc,'lat',numel(data.lat));
    dim_lon = netcdf.defDim(nc,'lon',numel(data.lon));
  end

  % variables:
  if addu
    var_u=netcdf.defVar(nc,'u','NC_DOUBLE',dim);
    netcdf.putAtt(nc,var_u,'long_name','reconstructed zonal wind');
  end
  if addv
    var_v=netcdf.defVar(nc,'v','NC_DOUBLE',dim);
    netcdf.putAtt(nc,var_v,'long_name','reconstructed meridional wind');
  end
  if addz
    var_z=netcdf.defVar(nc,'z','NC_DOUBLE',dim);
    netcdf.putAtt(nc,var_z,'long_name','reconstructed geopotential perturbation');
  end

  var_x=netcdf.defVar(nc,'lon','NC_DOUBLE',dim_lon);
  var_y=netcdf.defVar(nc,'lat','NC_DOUBLE',dim_lat);

  netcdf.putAtt(nc,var_x,'long_name','longitude');
  netcdf.putAtt(nc,var_y,'long_name','latitude');

  % global attributes:
  vid = netcdf.getConstant('GLOBAL');
  netcdf.putAtt(nc,vid,'date',datestr(now));
  f=fieldnames(attrs);
  for i=1:length(f)
    netcdf.putAtt(nc,vid,f{i},attrs.(f{i}));
  end

  netcdf.endDef(nc);

  % fill variables:
  if addu, netcdf.putVar(nc,var_u, permute(data.u,ndims(data.u):-1:1) );end
  if addv, netcdf.putVar(nc,var_v, permute(data.v,ndims(data.v):-1:1) );end
  if addz, netcdf.putVar(nc,var_z, permute(data.z,ndims(data.z):-1:1) );end
  netcdf.putVar(nc,var_x, permute(data.lon,ndims(data.lon):-1:1) );
  netcdf.putVar(nc,var_y, permute(data.lat,ndims(data.lat):-1:1) );
  netcdf.close(nc)
end
