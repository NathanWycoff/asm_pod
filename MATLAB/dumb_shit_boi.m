pkg load control

N = 30;
z = normrnd(0,1,2*N,1);

dumb_shit(z)

zopt = fminsearch(@dumb_shit, z)

dumb_shit(zopt)
