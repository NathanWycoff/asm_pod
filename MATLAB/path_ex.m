%  MATLAB/path_ex.m Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 05.06.2019
if (exist ("OCTAVE_VERSION", "builtin") > 0)
    pkg load control
end


% Try to create a system with no room for reducibility.
N = 10;
A = diag(repmat(-1,N,1));
B = repmat(1,N,1);
C = B';
D = 0;

sys = ss(A,B,C,D);
a = hsvd(sys);

gram(sys, 'o')
gram(sys, 'c')
