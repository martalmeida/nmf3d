% ------------------------------------ adjust paths:
addpath('../nmf3d_mat'); % change for your case
datafolder='./nmf3d_data/';

% ------------------------------------ vertical structure:
f=[datafolder 'T_ERA_I_1979_2010.txt'];
a=load(f);
T=a(1,:);
Lev=a(2,:);

[GnT,hkT,vfileT]=vertical_structure(T,Lev,'ws0',1);
[GnF,hkF,vfileF]=vertical_structure(T,Lev,'ws0',0);

% ------------------------------------ Hough functions:
M=6;
nLR=8;
nLG=6;
nk=5; % number of function to keep

[hvf_dataT,hfileT]=hough_functions(hkT(1:nk),M,nLR,nLG,'linear','dlat',6);
[hvf_dataF,hfileF]=hough_functions(hkF(1:nk),M,nLR,nLG,'linear','dlat',6);

% ------------------------------------ expansion coefficients:
% 3-D spectrum of total energy W_nlk
fu    = [datafolder 'u_01_1985_.nc4'];
fv    = [datafolder 'v_01_1985_.nc4'];
fz    = [datafolder 'z_01_1985_.nc4'];
fzref = [datafolder 'PHI_raw.txt'];

height=0;
data=load_ERA_I(fu,fv,fz,fzref,height);

[w_nlkT,wfileT]=expansion_coeffs(vfileT,hfileT,data,'label','outT');
[w_nlkF,wfileF]=expansion_coeffs(vfileF,hfileF,data,'label','outF');

% energy:
w_nlk=w_nlkT;
for i=1:nk
    E0(i,1,:,:)=1/4*1e5*hkT(i)*(w_nlk(i,1,:,:).*conj(w_nlk(i,1,:,:)));
    En(i,:,:,:)=1/2*1e5*hkT(i)*(w_nlk(i,2:end,:,:).*conj(w_nlk(i,2:end,:,:)));
end

% 3-D spectrum of energy interactions
if 0
  % assuming the coefficients were calculated and stored in the files
  % I1.mat, I2.mat and J3.mat; and the user has the variables lon, lat and P:

  data_i1=struct('lon',lon,'lat',lat,'P',P,'v',load('I1.mat'));
  data_i2=struct('lon',lon,'lat',lat,'P',P,'v',load('I2.mat'));
  data_j3=struct('lon',lon,'lat',lat,'P',P,'v',load('J3.mat'));

  data_i=struct('I1',data_i1,'I2',data_i2);
  [i_nlk,ifsave]=expansion_coeffs(vfileT,hfileT,data_i,'label','out_i_ws0_True');

  data_j=struct('J3',data_j3);
  [j_nlk,jfsave]=expansion_coeffs(vfileT,hfileT,data_j,'label','out_j_ws0_True');
end

% storing the vertical transform
if 0
  u=data.u.v;
  [u_k,fsave]=vertical_transform(u,hk,nk,Gn,p_old,p_new,dataLabel,'meridional wind')
end

% ------------------------------------ inverse expansion coefficients:
zi=[1,2,3];
mi=[1,2,3];
vi=[1,2];
pl=[850,500];
lon=0:30:359;
[uvz,invfsave]=inv_expansion_coeffs(vfileT,hfileT,wfileT,zi,mi,vi,pl,lon);
