function res=load_ERA_I(fu,fv,fz,fzref,height)
%
res=struct;
fprintf(1,'loading u : %s\n',fu);
res.('u')=load_ERA_I_once(fu);
fprintf(1,'loading v : %s\n',fv);
res.('v')=load_ERA_I_once(fv);
fprintf(1,'loading z : %s\n',fz);
res.('z')=load_ERA_I_once(fz,'fzref',fzref,'height',height);
end

function res=load_ERA_I_once(f,varargin)
  %

  fzref='';
  height=0;

  for i=1:length(varargin)
    if isequal(varargin{i},'fzref')
      fzref=varargin{i+1};
    elseif isequal(varargin{i},'height')
      height=varargin{i+1};
    end
  end

  nc=netcdf.open(f);
  lon=read_nc_var(nc,'lon');
  lat=read_nc_var(nc,'lat');
  P=read_nc_var(nc,'lev'); % pressure levels, Pa
  for i=netcdf.inqVarIDs(nc)
    [varname,xtype,dimids,natts]=netcdf.inqVar(nc,i);
    if length(dimids)==4
      fprintf(1,'    - loading %s\n',varname);
      v=netcdf.getVar(nc,i);
      v=permute(v,ndims(v):-1:1);
      break;
    end
  end
  netcdf.close(nc);

  % lat -90:90
  lat=lat(end:-1:1);
  v=v(:,:,end:-1:1,:);

  % subtracting reference:
  if fzref
    q=load(fzref);
    [c,n,ext]=fileparts(fzref);
    if isequal(ext,'.mat')
      q=q.phi0;
    else
      q=q(1,:);
    end
    fprintf(1,'    - subtracting reference\n');
    for i=1:length(P)
      v(:,i,:,:)=v(:,i,:,:)-q(i);
    end
  end

  if height
    constants;
    fprintf(1,'    - geopotential height --> geopotential\n');
    v=v/const.g;
  end

  res=struct;
  res.lon=lon;
  res.lat=lat;
  res.P=P;
  res.v=v;
end
