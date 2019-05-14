%  MATLAB/heat_ode.m Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 04.22.2019
pkg load control

load('../data/heatmodel.mat')

D = zeros(2,2);

sys = ss(A, B, C, D);

svs = hsvd(sys);
semilogy(svs(1:50)/svs(1), '.')
