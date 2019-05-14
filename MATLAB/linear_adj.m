%  MATLAB/linear_adj.m Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 04.23.2019

% Define problem
M = 10;
B = rand(M);
t_max = 1;

diff = @(t, x) B*x;

x0 = zeros([M,1]) + 1;
[ts, xs] = ode45(diff, [0, t_max], x0);
xsol = ode45(diff, [0, t_max], x0);

C = zeros([1,M]);
C(1) = 1;
plot(C*xs')

% Implement Adjoint Equation
lambda =  @(t) expm(-B'*t) \ (2*C'*C*deval(xsol, t));
