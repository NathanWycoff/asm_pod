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

function parameterized_lorenz(du,u,p,t)
    for n1 = 1:N
        du[n1] = 0;
        for n2 = 1:N
            du[n1] += p[n1]*A[n1, n2] * u[n2]
        end
    end
end

#u0 = [1.0,0.0,0.0]
u0 = zeros(N) .+ 1;
tspan = (0.0,1.0)
p = zeros(N) .+ 1;
prob = ODEProblem(parameterized_lorenz,u0,tspan,p)
sol = solve(prob)

g(u,p,t) = (sum(u.^2)) ./ 2

function dg(out,u,p,t)
    for n = 1:N
        out[n] = u[n]
    end
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
H = ForwardDiff.hessian(G,p)
