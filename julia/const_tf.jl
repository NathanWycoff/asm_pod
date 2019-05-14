using ControlSystems
using LinearAlgebra
using Random

N = 2;
#half_alphas = -abs(normrnd(0,1,N/2,1)) + normrnd(0,1,N/2,1)*1i;
#alphas = [half_alphas; conj(half_alphas)];
rng = MersenneTwister(1234);
#alphas = convert(Array{BigFloat,2}, -broadcast(abs, (randn(rng, Float64, (N, 1)))));
alphas = -broadcast(abs, (randn(rng, Float64, (N))));
A = convert(Array{Float64,2}, Diagonal(alphas));
B = zeros(N,1) .+ 1;
D = 0;

# Calculate the Ci's to make the system unitary
lambdas = alphas;

#logC = zeros(1,N);
rho = 1.0;
C = zeros(1,N);
for i = 1:N
    li = lambdas[i];
    num_vals = conj(lambdas) .+ li;
    denom_vals = li .- lambdas;
    deleteat!(denom_vals, i);
    C[1,i] = rho * prod(num_vals) / prod(denom_vals);
    #num_vals = log(conj(lambdas) .+ li);
    #denom_vals = log(li .- lambdas);
    #denom_vals(i) = [];
    #logC(1,i) = sum(num_vals) - sum(denom_vals);
end

#logC = logC - (max(real(logC)) + max(imag(logC)) * 1i)
#C = exp(logC);

sys = ss(A, B, C, D);
sigmas = hsvd(sys)
