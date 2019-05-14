using MAT
using DifferentialEquations
using ForwardDiff
using Plots
using DiffEqSensitivity
using Calculus
using QuadGK
using Sundials, DiffEqBase
using Plots

#########################################

matvars = matread("./data/heatmodel.mat");
At = matvars["A"]';
Bt = matvars["B"]';
Ct = matvars["C"]';

# build fourrier basis
K = 2;
sigmas = exp10.(range(-5.0, stop=2.0, length=K));

P = size(Bt)[1];
N = size(At)[1];

#function fourrier_basis(t)
#    mat = zeros((K,P*K))
#    mat[1,1:K] = sin.(sigmas.*t)
#    mat[2,(K+1):2K] = sin.(sigmas.*t)
#    return(mat)
#end

function fourrier_basis(t)
    return(sin.(sigmas.*t));
end

function f(du,u,p,t)
    du[:] = transpose(u)*At + transpose(transpose(p)*fourrier_basis(t))*Bt;
end

u0 = zeros(N) .+ 1;
tspan = (0.0,1.0)
p = zeros(K,P) .+ 1;
prob = ODEProblem(f,u0,tspan,p)
sol = solve(prob)

g(u,p,t) = (sum(u.^2)) ./ 2

function dg(out,u,p,t)
    for n = 1:N
        out[n] = u[n];
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
