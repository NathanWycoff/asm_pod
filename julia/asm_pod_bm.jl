using MAT
using ControlSystems
using DifferentialEquations
using ForwardDiff
using Plots
using Plots.PlotMeasures
using DiffEqSensitivity
using QuadGK
using LinearAlgebra
using ProgressMeter
using Optim
include("julia/heatmodel_lib.jl")

matvars = matread("./data/beam_model.mat");
A = matvars["A"];
B = matvars["B"];
C = matvars["C"];

# Change to second element of C
C = zeros(1,N);
C[2] = 1.;

#TODO: In many places, we assume t_max to be 1.
t_max = 1.0;

# build fourrier basis
K = 20;
sigmas = exp10.(range(-5.0, stop=2.0, length=K));

P = size(B)[2];
N = size(A)[1];

true_p = rand(K*P);
true_sol = solve_ode(true_p, A, B);
fig = plot(true_sol, vars = 2, left_margin = 10mm);
size(10,10);
savefig("temp.pdf")

p = rand(K*P);
obj_grad(p, A, B, C, true_sol, use_adj = false)

#@time obj_grad(p, use_adj = true)
#@time obj_grad(p, use_adj = true)
#@time obj_grad(p, use_adj = false)
#@time obj_grad(p, use_adj = false)

# Determine the Active Subspace.
M = convert(Int64, ceil(10*log(K)));# Number of AS MC samples.
grads = Array{Float64}(undef, M, K);
@showprogress for m = 1:M
    p = rand(K*P);
    grads[m,:] = obj_grad(p, A, B, C, true_sol, use_adj = false);
end

# Examine the singular values
Fg = svd(grads);
plot(Fg.S./maximum(Fg.S), yaxis=:log, seriestype=:scatter)

# The low D space we choose to keep in terms of the input space; gives number of POD runs.
L = 11;
FVr = Fg.V[:,1:L];

# Run the model with the given forcing terms.
Xs = [];
for l = 1:L
    p = FVr[:,l];

    odesol = solve_ode(p, A, B);
    t_steps = length(odesol.t);

    # Each column is a state, each row is a time step.
    X = Array{Float64}(undef, t_steps, N);

    for t = 1:t_steps
        X[t,:] = odesol.u[t];
    end

    push!(Xs, X);
end

# Perform POD on the entire enchilada
Xa = Xs[1];
for l = 2:L
    global Xa
    Xa = vcat(Xa, Xs[l]);
end

Fx = svd(Xa);
plot(Fx.S/maximum(Fx.S), yaxis=:log, seriestype=:scatter)
R = 50; # Dimension of underlying model.

# Define our reduced model
V = Fx.V[:,1:R];
Atr = V'*At*V;
Btr = Bt*V;
Ctr = V'*Ct;

# TODO: Next: We should make a general LTI system.

