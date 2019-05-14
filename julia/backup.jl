using MAT
using DifferentialEquations
using ForwardDiff
using Plots
using DiffEqSensitivity
using Calculus
using QuadGK
using Sundials, DiffEqBase

matvars = matread("./data/heatmodel.mat");
A = matvars["A"];
B = matvars["B"];
C = matvars["C"];

N = size(A)[1];

function f(du,u,p,t)
    dstate = A*u + B*p;
    for n = 1:N
        du[n] = dstate[n];
    end
end

u0 = zeros(N);
tspan = (0.0,1.0)
p = [1,1];
prob = ODEProblem(f,u0,tspan,p)
sol = solve(prob)

plot(sol, vars = (0,100))

# Sensitivity wrt input
g(u,p,t) = (sum(u).^2) ./ 2;
# dg(u,p,t) = u;
s = adjoint_sensitivities(sol,Vern9(),g,nothing)

#########################################
function parameterized_lorenz(du,u,p,t)
 du[1] = p[1]*(u[2]-u[1])
 du[2] = u[1]*(p[2]-u[3]) - u[2]
 du[3] = u[1]*u[2] - p[3]*u[3]
end

u0 = [1.0,0.0,0.0]
tspan = (0.0,1.0)
p = [10.0,28.0,8/3]
prob = ODEProblem(parameterized_lorenz,u0,tspan,p)
sol = solve(prob)

g(u,p,t) = (sum(u.^2)) ./ 2

function dg(out,u,p,t)
  #out[1]= u[1] + u[2]
  #out[2]= u[1] + u[2]
  out[1] = u[1]
  out[2] = u[2]
  out[3] = u[3]
end

res = adjoint_sensitivities(sol,Vern9(),g,nothing,dg,abstol=1e-8,
                                 reltol=1e-8,iabstol=1e-8,ireltol=1e-8)

function G(p)
    tmp_prob = remake(prob;u0=convert.(eltype(p),prob.u0),p=p)
    sol = solve(tmp_prob,Vern9(),abstol=1e-14,reltol=1e-14)
    res,err = quadgk((t)-> (sum(sol(t).^2))./2,0.0,1.0,abstol=1e-14,reltol=1e-10)
    res
end
res2 = ForwardDiff.gradient(G,p)
Hλ = ForwardDiff.hessian(G,p)
