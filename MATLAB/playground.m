%  MATLAB/playground.m Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 04.22.2019

% Define the problem
load('../data/heatmodel.mat')
[M,M] = size(A);
[M,P] = size(B);
[Q,M] = size(C);
t_max = 1;
K = 10;% Size of basis for input discretization in Fourrier space.

% Form the input, plot it for fun
sigma = logspace(-5,1, K);
form_coefs = @(t) [sin(sigma.*t) zeros([1,K]); zeros([1,K]) sin(sigma.*t)];
alpha = rand([K*P,1]);

u = @(t) (form_coefs(t) * alpha);
us = zeros([100,2]);
for t = 1:100
    us(t,:) = u(t/100);
end
plot(us)

% The source equation
heat_diff = @(t, x) A * x + B * u(t);

% Specify intial condition and solve equation.
x0 = rand([M,1]);
[ts,xs] = ode45(heat_diff, [0, t_max], x0);
xsol = ode45(heat_diff, [0, t_max], x0);

% Observe response.
ys = C * xs';
plot(ys')

% Solve the Adjoint equation, note that this needs to be backwards in time.
adj_diff = @(t, lambda) -A'*lambda + 2*deval(xsol, t_max - t)
lambda0 = zeros([M,1]);
[ts,lambdas] = ode45(adj_diff, [0, t_max], lambda0);
lsol = ode45(adj_diff, [0, t_max], lambda0);

lys = C * lambdas';
plot(lys');

% Compute the gradient given the calculated quantities.
integrand = @(t) deval(lsol, t_max - t)'*B*form_coefs(t);

for ki = 1:(K*P)
    ei = zeros([K*P,1]);
    ei(ki) = 1;
    integrandi = @(t) integrand(t)*ei;
    integral(integrandi, 0, t_max)
end
