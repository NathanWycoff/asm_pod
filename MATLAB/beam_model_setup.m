% 
if (exist ("OCTAVE_VERSION", "builtin") > 0)
    pkg load control
end
set(0, 'defaultaxesfontsize',14,'defaultaxeslinewidth',1.0,...
    'defaultlinelinewidth',2.0,'defaultpatchlinewidth',1.0,...
    'defaulttextfontsize',18,'DefaultLineMarkerSize',14);


nf = 200;  
[M,K]=finbeam(80,nf,2700,.00651*.02547,.00651^3*.02547/12,7.1e10);
M  = sparse(M); K  = sparse(K);
% Choose damping parameters
alpha = 1/10000; beta  = 1/10;  
% Damping matrix
D =  alpha*K + beta*M; 


n2 = size(M,2);  % Dimension in the second-order, i.e.,
                 % H(s) = C2*(s^2M + s*D + K)^{-1}*B2, framework
B2 = eye(n2,1); C2 = eye(n2,1)'; 

% If desired, can be convered to E,A,B,C form
n = 2*n2;  % Dimension in the first-order H(s) = C*(s*E - A)^{-1}*B framework

A = full([zeros(n2) speye(n2) ; -K -D]);
B = [zeros(n2,1); B2];
C = eye(n,1)'; 
D = 0;
E = full([eye(n2) zeros(n2) ; zeros(n2) M]);
% Get rid of E for simplicity
esize = size(E, 1);
E  = eye(esize);

sys = dss(A,B,C,D,E);

svs = hsvd(sys);
semilogy(svs/svs(1))

save('../data/beam_model.mat', 'A','B','C','D','E')
