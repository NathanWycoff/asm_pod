using MAT
using DifferentialEquations
using ForwardDiff
using Plots
using DiffEqSensitivity

using Calculus
using QuadGK
using Sundials, DiffEqBase
#using Plots

#########################################

matvars = matread("./data/heatmodel.mat");
At = matvars["A"]';
Bt = matvars["B"]';
Ct = matvars["C"]';


# build fourrier basis
K = 3;
sigmas = exp10.(range(-5.0, stop=2.0, length=K));

P = size(Bt)[1];
N = size(At)[1];

# Smaller problem for debugging
# TODO: Remove
#N = 10;
#At = At[1:N, 1:N];
#Bt = Bt[:,1:N];
#Ct = Ct[1:N,:];

#function fourrier_basis(t)
#    mat = zeros((K,P*K))
#    mat[1,1:K] = sin.(sigmas.*t)
#    mat[2,(K+1):2K] = sin.(sigmas.*t)
#    return(mat)
#end

function fourrier_basis(t)
    return(sin.(sigmas.*t));
end
function vecfourrier_basis(t)
    return([sin.(sigmas.*t) ; sin.(sigmas.*t)]);
end

function f(du,u,p,t)
    #du[:] = transpose(u)*At + transpose(transpose(p)*fourrier_basis(t))*Bt;
    vfb = vecfourrier_basis(t);
    du[:] = transpose(u)*At + transpose(transpose(p[1:K])*vfb[1:K])*Bt[1,:]' + transpose(transpose(p[(K+1):(2*K)])*vfb[(K+1):(2*K)])*Bt[2,:]';
end

function adj_grad(p)
    u0 = zeros(N) .+ 1;
    u0 = convert.(eltype(p),u0)
    tspan = (0.0,1.0)
    prob = ODEProblem(f,u0,tspan,p)
    sol = solve(prob)

    g(u,p,t) = convert.(eltype(p), (sum(u.^2)) ./ 2);

    function dg(out,u,p,t)
        out[:] = convert.(eltype(p), u);
    end

    #res = adjoint_sensitivities(sol,Vern9(),g,nothing,dg,abstol=1e-8,
    #                                 reltol=1e-8,iabstol=1e-8,ireltol=1e-8)
    res = adjoint_sensitivities(sol,Vern9(),g,nothing,dg,abstol=1e-8,
                                     reltol=1e-8,iabstol=1e-8,ireltol=1e-8,
                                    sensealg=SensitivityAlg(autodiff=false))
    return(vec(res))
end

p = rand(K*P);
@time adj_grad(p)

@time ForwardDiff.jacobian(adj_grad, p)

function G(p)
    tmp_prob = remake(prob;u0=convert.(eltype(p),prob.u0),p=p)
    sol = solve(tmp_prob,Vern9(),abstol=1e-14,reltol=1e-14)
    res,err = quadgk((t)-> (sum(sol(t).^2))./2,0.0,1.0,abstol=1e-14,reltol=1e-10)
    res
end
H = ForwardDiff.hessian(G,p)
H = Calculus.hessian(G,p)
