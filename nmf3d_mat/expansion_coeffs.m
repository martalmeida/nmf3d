function varargout=expansion_coeffs(vfile,hfile,data,varargin)
%  Expansion coefficients
%  Compute the Vertical, Fourier and Hough transforms of:
%    - zonal wind, meridional wind and geopotential perturbation (from the
%      reference  geopotential), used for the the 3-D spectrum of total energy
%      W_nlk
%    - I1, I2 and J3 (see vertical_transform), used for the 3-D spectrum of
%      energy interactions (kinetic and available pontential energy)
%
%  vfile: equivalent heights and vertical structure functions
%  hfile: Hough functions
%  data: structure with fields (u,v,z) or (I1,I2) or (J3)
%  varargin:
%    save, create file [true]
%    format, file format: [nc] or mat
%    attrs, attributes to save [{}]
%    label, start of the saved filename ['out']
%
%  Returns the expansion coefficients (eddies and zonal components combined),
%  as well as saved filename if save is true

constants;

save=true; % create file
for i=1:length(varargin)
  if isequal(varargin{i},'save')
    save=varargin{i+1};
  end
end

fprintf(1,'- Expansion coefficients -\n')
fprintf(1,' - loading parameters from Hough functions file:\n    %s\n',hfile);
if ends_with(hfile,'.mat')
  ohfile=load(hfile);
  [three,nN,nmm,nk,nLat]=size(ohfile.HOUGHs_UVZ);
else % netcdf
  % loading dimensions:
  nc=netcdf.open(hfile);
  [name,nN]=netcdf.inqDim(nc,netcdf.inqDimID(nc,'max_zonal_wave_number_and_zonal_mean'));
  [name,nk]=netcdf.inqDim(nc,netcdf.inqDimID(nc,'number_equivalent_heights'));
  netcdf.close(nc);
end

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


if isfield(data,'u') % zonal wind, meridional wind, geopotential
  a_nk=prep(data.('u'),p_new,nk,Gn,hk,nN,'zonal wind');
  b_nk=prep(data.('v'),p_new,nk,Gn,hk,nN,'meridional wind');
  c_nk=prep(data.('z'),p_new,nk,Gn,hk,nN,'geopotential');
  var_nk = zeros([3,size(a_nk)]); % complex

  % Storing vertical and Fourier transforms of u, v and PHI:
  var_nk(1,:,:,:,:)=a_nk;
  var_nk(2,:,:,:,:)=b_nk;
  var_nk(3,:,:,:,:)=c_nk;

  Lat=data.('u').('lat');
  type='uvz';

elseif isfield(data,'I1')
  a_nk=prep(data.('I1'),p_new,nk,Gn,hk,nN,'I1');
  b_nk=prep(data.('I2'),p_new,nk,Gn,hk,nN,'I2');
  var_nk = zeros([3,size(a_nk)]); % complex

  var_nk(1,:,:,:,:)=a_nk;
  var_nk(2,:,:,:,:)=b_nk;

  Lat=data.('I1').('lat');
  type='I';

elseif isfield(data,'J3')
  c_nk=prep(data.('J3'),p_new,nk,Gn,hk,nN,'J3');
  var_nk = zeros([3,size(c_nk)]); % complex

  var_nk(3,:,:,:,:)=c_nk;

  Lat=data.('J3').('lat');
  type='J';
end


% Hough transforms -------------------------------------------------
fprintf(1,' - loading Hough vector functions:\n    %s\n',hfile);
if ends_with(hfile,'.mat')
  HOUGHs_UVZ = ohfile.HOUGHs_UVZ;
else
  nc=netcdf.open(hfile);
  HOUGHs_UVZ=read_nc_var(nc,'HOUGHs_UVZ_real')+1j*read_nc_var(nc,'HOUGHs_UVZ_imag');
  netcdf.close(nc);
end

var_nlk=hough_transform(HOUGHs_UVZ,var_nk,Lat,ws0);

if save
  if type=='uvz'
    data=struct('w_nlk',var_nlk);
  elseif type=='I'
    data=struct('i_nlk',var_nlk);
  elseif type=='J'
    data=struct('j_nlk',var_nlk);
  end

  fsave=save_out(data,varargin{:});
  varargout={var_nlk,fsave};
else
  varargout={var_nlk};
end

end


function U_nk=prep(data,p_new,nk,Gn,hk,nN,dataLabel)
  constants;

  u   = data.v;
  Lat = data.lat;
  Lon = data.lon;
  P   = data.P;

  u_k=vertical_transform(u,hk,nk,Gn,P,p_new,dataLabel);

  fprintf(1,' - %s - Fourier transform\n',dataLabel);
  % Fourier transform
  U_nk = fft(u_k,[],3);    % U_nk is the Fourier Transform of u_k along dimension 3 (i.e along longitude)
  U_nk = U_nk/length(Lon); % Scale the fft so that it is not a function of the length of input vector

  % Retaining the first nN (see hough_functions) zonal wave numbers
  % U_nk has dimensions U_nk(m,Lat,nN=Lon,t)
  U_nk = U_nk(:,:,1:nN,:);   % First nN wavenumbers with zonal mean

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

  if     isfield(data,'w_nlk'), label2='w';
  elseif isfield(data,'i_nlk'), label2='i';
  elseif isfield(data,'j_nlk'), label2='j';
  end

  fsave=sprintf('%s_%s_nlk.%s',label,label2,format);

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

  if isfield(data,'w_nlk')
    vname='w_nlk';
    vlname='Expansion coefficients of dependent variable vector u, v, z';
  elseif isfield(data,'i_nlk')
    vname='i_nlk';
    vlname='Expansion coefficients of nonlinear term vector due to wind field';
  elseif isfield(data,'j_nlk')
    vname='j_nlk';
    vlname='Expansion coefficients of nonlinear term due to mass field';
  end
  v=data.(vname);

  % dimensions:
  [nk,nN,nL,nTime]=size(v);
  dim_nk=netcdf.defDim(nc,'number_equivalent_heights',nk);
  dim_nN=netcdf.defDim(nc,'max_zonal_wave_number',nN);
  dim_nL=netcdf.defDim(nc,'total_meridional_modes',nL); % LG+LR
  dim_nT=netcdf.defDim(nc,'time',nTime);

  % variables:
  dim=[dim_nk,dim_nN,dim_nL,dim_nT];
  dim=dim(end:-1:1);
  var_r=netcdf.defVar(nc,[vname '_real'],'NC_DOUBLE',dim);
  netcdf.putAtt(nc,var_r,'long_name',[vlname ' (real)']);
  var_i=netcdf.defVar(nc,[vname '_imag'],'NC_DOUBLE',dim);
  netcdf.putAtt(nc,var_i,'long_name',[vlname ' (imag)']);

  % global attributes:
  vid = netcdf.getConstant('GLOBAL');
  netcdf.putAtt(nc,vid,'date',datestr(now));
  f=fieldnames(attrs);
  for i=1:length(f)
    netcdf.putAtt(nc,vid,f{i},attrs.(f{i}));
  end

  netcdf.endDef(nc);

  % fill variables:
  netcdf.putVar(nc,var_r,  permute(real(v),ndims(v):-1:1) );
  netcdf.putVar(nc,var_i,  permute(imag(v),ndims(v):-1:1) );
  netcdf.close(nc);
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
