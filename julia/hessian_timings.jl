using MAT
using JLD
using DifferentialEquations
using ForwardDiff
using Plots
using DiffEqSensitivity
using Calculus
using QuadGK
using Statistics

## Compare different methods for obtaining a hessian:
# Full forward AD
# Forward over reverse AD
# Full FD
# FD over reverse AD
n_trials = 10

matvars = matread("./data/heatmodel.mat");
At = matvars["A"]';
Bt = matvars["B"]';
Ct = matvars["C"]';

#TODO: In many places, we assume t_max to be 1.
t_max = 1;

# build fourrier basis
K = 200
sigmas = exp10.(range(-5.0, stop=2.0, length=K));

P = size(Bt)[1];
N = size(At)[1];

function fourrier_basis(t)
    return(sin.(sigmas.*t));
end
function vecfourrier_basis(t)
    return([sin.(sigmas.*t) ; sin.(sigmas.*t)]);
end

function control(t, p)
    p[1:K]' * fourrier_basis(t) + p[(K+1):(K*P)]'* fourrier_basis(t)
end

function f(du,u,p,t)
    vfb = vecfourrier_basis(t);
    du[:] = transpose(u)*At + transpose(transpose(p[1:K])*vfb[1:K])*Bt[1,:]' + transpose(transpose(p[(K+1):(2*K)])*vfb[(K+1):(2*K)])*Bt[2,:]';
end

function solve_ode(p)
    u0 = zeros(N) .+ 1;
    u0 = convert.(eltype(p),u0)
    tspan = (0.0,1.0)
    prob = ODEProblem(f,u0,tspan,p)
    sol = solve(prob)
    return sol
end

true_p = rand(K*P);
true_sol = solve_ode(true_p);

function adj_grad(p)
    sol = solve_ode(p);

    g(u,p,t) = convert.(eltype(p), (sum((u - true_sol(t)).^2)) ./ 2);

    function dg(out,u,p,t)
        out[:] = convert.(eltype(p), u - true_sol(t));
    end

    res = adjoint_sensitivities(sol,Vern9(),g,nothing,dg,abstol=1e-8,
                                     reltol=1e-8,iabstol=1e-8,ireltol=1e-8,
                                    sensealg=SensitivityAlg(autodiff=false))
    return(vec(res))
end

function G(p)
    #tmp_prob = remake(prob;u0=convert.(eltype(p),prob.u0),p=p)
    sol = solve_ode(p);
    #sol = solve(tmp_prob,Vern9(),abstol=1e-14,reltol=1e-14)
    res,err = quadgk((t)-> (sum((sol(t) - true_sol(t)).^2))./2,0.0,1.0,atol=1e-14,rtol=1e-10)
    return(res)
end

function fd_over_rev(p)
    h = 1e-6;
    base = adj_grad(p);
    Jac = zeros(K*P,K*P);
    for i = 1:(P*K)
        ph = zeros(K*P);
        ph[:] = p;
        ph[i] += h;
        step = adj_grad(ph);
        Jac[:,i] = (step - base) ./ h;
    end
    return(Jac)
end

###### Analytic stuff
#function as_int(t,p)
#    exp(At'*(t_max - t)) * Bt'* control(t,p) 
#end
#
#function anlyt_sol(p)
#    res,err = quadgk((t)-> as_int(t, p),0.0,1.0,atol=1e-14,rtol=1e-10)
#    return(res)
#end
#
#function grad_integrand(t,p)
#    exp(A(1-t))
#end
#
#function anlyt_grad(p)
#    res,err = quadgk((t)-> p./2,0.0,1.0,atol=1e-14,rtol=1e-10)
#end
##### Analytic stuff

Ks = [100 200 500];
solve_times = zeros(n_trials, length(Ks)) .- 1;

rev_grad_times = zeros(n_trials, length(Ks)) .- 1;
fd_grad_times = zeros(n_trials, length(Ks)) .- 1;
for_grad_times = zeros(n_trials, length(Ks)) .- 1;

for_for_times = zeros(n_trials, length(Ks)) .- 1;
for_rev_times = zeros(n_trials, length(Ks)) .- 1;
fd_fd_times = zeros(n_trials, length(Ks)) .- 1;
fd_rev_times = zeros(n_trials, length(Ks)) .- 1;

ki = 0;
for Ka = Ks
    global K;
    # I don't understand julia
    K = Ka;
    global ki;
    global sigmas;
    ki = ki + 1;
    sigmas = exp10.(range(-5.0, stop=2.0, length=K));
    for tri = 1:n_trials
        print(tri, "\n")
        p = rand(K*P);

        # Timing calculation of the objective.
        a1 = @timed G(p);
        solve_times[tri,ki] = a1[2];

        # Timing gradient methods
        a2 = @timed adj_grad(p);
        rev_grad_times[tri,ki] = a2[2];

        a3 = @timed Calculus.gradient(G, p);
        fd_grad_times[tri,ki] = a3[2];

        a4 = @timed ForwardDiff.gradient(G, p);
        for_grad_times[tri,ki] = a4[2];

        # Hessians
        #a5 = @timed ForwardDiff.hessian(G, p);
        #for_for_times[tri] = a5[2];

        #a6 = @timed ForwardDiff.jacobian(adj_grad, p);
        #for_rev_times[tri] = a6[2];

        #a7 = @timed Calculus.hessian(G, p);
        #fd_fd_times[tri] = a7[2];

        #a8 = @timed fd_over_rev(p);
        #fd_rev_times[tri] = a8[2];
    end
end

for ki = 1:length(K)
    print(ki)
    print(median(rev_grad_times[:,ki]), "\n")
    print(median(fd_grad_times[:,ki]), "\n")
    print(median(for_grad_times[:,ki]), "\n")
end


save("./data/hessian_timings.jld", "solve_times", solve_times, "rev_grad_times", rev_grad_times, \
     "fd_grad_times", fd_grad_times, "for_grad_times", for_grad_times, "for_for_times", for_for_times, \
     "for_rev_times", for_rev_times, "fd_fd_times", fd_fd_times, "fd_rev_times", fd_rev_times);
