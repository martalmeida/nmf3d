function v=read_nc_var(nc,name)
  if ischar(nc)
    nc=netcdf.open(nc);
  end

  i=netcdf.inqVarID(nc,name);
  v=netcdf.getVar(nc,i);

  % permute if more than one dim:
  [varname,xtype,dimids,natts]=netcdf.inqVar(nc,i);
  nd=length(dimids);
  if nd>1
    v=permute(v,ndims(v):-1:1);
  end

end

