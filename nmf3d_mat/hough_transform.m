function var_nlk=hough_transform(HOUGHs_UVZ,var_nk,Lat,ws0)
%  Hough transform
%
%  HOUGHs_UVZ, Hough functions
%  var_nk, vertical and Fourier transforms of (u,v,z) or (I1,I2) or (J3) (see vertical_transform)
%  Lat, latitudes of the data and Houghs
%  ws0, if true the pressure vertical velocity is zero at surface


% check if linear or gaussian
if length(unique(diff(Lat)))==1 % linear
  Dl  = (Lat(2)-Lat(1))*pi/180; % Latitude spacing (radians)
  latType='linear';
else % gaussian
  [tmp,gw]=lgwt(length(Lat),-1,1);
  latType='gaussian'
end

% Eliminate geopotential/J3 (used only for reconstruction) if ws0:
if ws0
  HOUGHs_UVZ(3,:,:,1,:)=0;
end

% dimensions:
[three,nM,nL,nk,nLat]=size(HOUGHs_UVZ);
nTimes=size(var_nk,5);

disp(' - computing')
%cosTheta = repmat(cos(Lat'*pi/180),[3 1]);
cosTheta = repmat(cosd(Lat'),[3 1]);
var_nlk = zeros(nk,nM,nL,nTimes); % complex
for k=1:nk % vertical index
  for n=1:nM % wavenumber index
    for l=1:nL % meridional index
      for t=1:nTimes % time index
        Aux=squeeze(var_nk(:,k,:,n,t)).*squeeze(conj(HOUGHs_UVZ(:,n,l,k,:))).*cosTheta; % Aux(3,Lat)
        y1=sum(Aux); % Integrand -> y1(Lat)

        if isequal(latType,'linear')
          % Computes latitude integral of the Integrand using trapezoidal method
          aux1=(y1(1:end-1)+y1(2:end))*Dl/2.; % len Lat.size-1; complex
        elseif isequal(latType,'gaussian')
          aux1=gw*y1;
        end

        var_nlk(k,n,l,t) = sum(aux1);

      end
    end
  end
end
