#TODO: Work on making sure A,B,C are never the global ones.

function matfourrier_basis(t)
    #return([sin.(sigmas.*t) zeros(K); zeros(K) sin.(sigmas.*t)]');
    return(sin.(sigmas.*t)');
end

function solve_ode(p, A, B)
    N = size(A)[1];
    function f(du,u,p,t)
        mfb = matfourrier_basis(t);
        du[:] = transpose(A*u + B*mfb*p)
    end
    u0 = zeros(N);
    u0 = convert.(eltype(p),u0)
    tspan = (0.0,t_max)
    prob = ODEProblem(f,u0,tspan,p)
    sol = solve(prob)
    return sol
end

function adj_grad(p, C)
    sol = solve_ode(p, A, B);

    g(u,p,t) = convert.(eltype(p), (sum((C*u - C*true_sol(t)).^2)) ./ 2);

    function dg(out,u,p,t)
        #TODO: This C outer product is highly inefficient
        out[:] = convert.(eltype(p), C'*(C*(u - true_sol(t))));
    end

    res = adjoint_sensitivities(sol,Vern9(),g,nothing,dg,abstol=1e-8,
                                     reltol=1e-8,iabstol=1e-8,ireltol=1e-8,
                                    sensealg=SensitivityAlg(autodiff=false))
    return(vec(res))
end

function obj(p, A, B, C)
    #tmp_prob = remake(prob;u0=convert.(eltype(p),prob.u0),p=p)
    sol = solve_ode(p, A, B);
    #sol = solve(tmp_prob,Vern9(),abstol=1e-14,reltol=1e-14)
    res,err = quadgk((t)-> (sum((C*sol(t).-1).^2))./2,0.0,1.0,atol=1e-14,rtol=1e-10)
    return(res)
end

# p The param vector which we want a gradient from
# use_adj If true, do adjoint method (reverse mode AD); if false, use forward mode AD.
function obj_grad(p, A, B, C; use_adj = false)
    if use_adj
        return(adj_grad(p, C))
    else
    function G(p)
        #tmp_prob = remake(prob;u0=convert.(eltype(p),prob.u0),p=p)
        sol = solve_ode(p, A, B);
        #sol = solve(tmp_prob,Vern9(),abstol=1e-14,reltol=1e-14)
        #res,err = quadgk((t)-> (sum((C*(sol(t) - true_sol(t))).^2))./2,0.0,1.0,atol=1e-14,rtol=1e-10)
        res,err = quadgk((t)-> (sum((C*sol(t).-1).^2))./2,0.0,1.0,atol=1e-14,rtol=1e-10)
        return(res)
    end

        return(ForwardDiff.gradient(G, p))
    end
end
