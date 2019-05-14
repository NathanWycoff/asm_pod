pkg load control
pkg load signal

N = 30;
%half_alphas = -abs(normrnd(0,1,N/2,1)) + normrnd(0,1,N/2,1)*1i;
%alphas = [half_alphas; conj(half_alphas)];
%alphas = -abs(normrnd(0,1,N,1)) + normrnd(0,1,N,1)*1i
%alphas = -abs(normrnd(0,1,N,1)) 
%alphas = -abs(normrnd(0,1,N,1)) 
raw_alphas = -(1:N);
alphas = raw_alphas / N - 10
A = diag(alphas);
B = zeros(N,1) .+ 1;
D = 0;

% Calculate the Ci's to make the system unitary
lambdas = vpa(sym(raw_alphas), 32) / sym(N) + sym(10);

logC = zeros(1,N);
rho = 1.0;
C = sym(zeros(1,N));
for i = 1:N
    li = lambdas(i);
    % Method 1
    num_vals = conj(lambdas) .+ li;
    denom_vals = li .- lambdas;
    denom_vals = [denom_vals(1:i-1) denom_vals((i+1):end)];
    C(1,i) = rho * prod(num_vals) / prod(denom_vals);
end

C = C / max(abs(C))
C = double(C)



    %% Method 2
    %num_vals = conj(lambdas) .+ li;
    %denom_vals = li .- lambdas;
    %C(1,i) = rho*num_vals(i);
    %for j = 1:N
    %    if j == i
    %        continue
    %    end
    %    C(1,i) *= num_vals(j) / denom_vals(j);
    %end

    % Method 3
    %num_vals = log(conj(lambdas) .+ li);
    %denom_vals = log(li .- lambdas);
    %denom_vals(i) = [];
    %logC(1,i) = sum(num_vals) - sum(denom_vals);

%logC = logC - (max(real(logC)) + max(imag(logC)) * 1i)
%logC = logC - (max(real(logC)))
%C = exp(logC);

sys = ss(A, B, C, D);
sigmas = hsvd(sys)
