% adjust paths: ------------------------------------------------------
d='SOME_PATH/nmf3d/'
addpath([d 'nmf3d_mat']);
datafolder='nmf3d_data/';

% vertical structure: ------------------------------------------------
f=[datafolder 'T_ERA_I_1979_2010.txt'];
a=load(f);
T=a(1,:);
Lev=a(2,:);
[Gn,hk,vfile]=vertical_structure(T,Lev,'ws0',0);

% hough functions: ---------------------------------------------------
M=6;
nLR=8;
nLG=6;
nk=5; % number of function to keep

% ws0=1
[Gn,hk,vfileT]=vertical_structure(T,Lev,'ws0',1);
[hvf_dataT,hfileT]=hough_functions(hk(1:nk),M,nLR,nLG,'linear','dlat',6);

% ws0=0
[Gn,hk,vfileF]=vertical_structure(T,Lev,'ws0',0);
[hvf_dataF,hfileF]=hough_functions(hk(1:nk),M,nLR,nLG,'linear','dlat',6);

% Expansion coefficients: --------------------------------------------
% 3-D spectrum of total energy W_nlk
fu=[datafolder 'u_01_1979_.nc4'];
fv=[datafolder 'v_01_1979_.nc4'];
fz=[datafolder 'z_01_1979_.nc4'];
fzref=[datafolder 'PHI_raw.txt'];
height=0;
data=load_ERA_I(fu,fv,fz,fzref,height);

% ws0=1
[w_nlk,fsave]=expansion_coeffs(vfileT,hfileT,data,'label','out_ws0_True');

% ws0=0
[w_nlk,fsave]=expansion_coeffs(vfileF,hfileF,data,'label','out_ws0_False');


for i=1:nk
    E0(i,1,:,:)=1/8*1e5*hk(i)*(w_nlk(i,1,:,:).*conj(w_nlk(i,1,:,:)));
    En(i,:,:,:)=1/4*1e5*hk(i)*(w_nlk(i,2:end,:,:).*conj(w_nlk(i,2:end,:,:)));
end


% 3-D spectrum of energy interactions
data_i1=struct;
data_i1.lon=lon;
data_i1.lat=lat;
data_i1.P=P;
data_i1.v=load('I1.mat');

data_i2=struct;
data_i2.lon=lon;
data_i2.lat=lat;
data_i2.P=P;
data_i2.v=load('I2.mat');

data_j3=struct;
data_j3.lon=lon;
data_j3.lat=lat;
data_j3.P=P;
data_j3.v=load('J3.mat');

data_i=struct;
data_i.I1=data_i1;
data_i.I2=data_i2;
[i_nlk,ifsave]=expansion_coeffs(vfileT,hfileT,data_i,'label','out_i_ws0_True');

data_j=struct;
data_j.J3=data_j3;
[j_nlk,jfsave]=expansion_coeffs(vfileT,hfileT,data_j,'label','out_j_ws0_True');
