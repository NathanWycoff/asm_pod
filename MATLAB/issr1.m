pkg load control
load('../data/iss1R_matrices.mat')

D = zeros(270,3);
%C = zeros(1,270);
%C(1,1) = 1;
C = eye(270);

sys = ss(A,B,C,D);
P = gram(sys, 'c');
Q = gram(sys, 'o');

hsvd(sys)
