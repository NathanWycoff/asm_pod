function [M,K]=finbeam(l,numel,rou,A,I,E)
%[M,K,S,FreqH,P]=finbeam(l,numel,rou,A,I,E)
%
% This program generates the a Finite Element Model for an Euler-Bernoulli 
% of a Cantilever beam
% given the inputs
%                   l       The length of the beam.
%                   numel   The number of elements desired
%                   rou     The density of the material
%                   A       The cross sectional area
%                   I       moment of inertia
%                   E       Young's Modulus(Elastic Modulus)
% The outputs are
%                   M       The Mass Matrix
%                   K       The stiffness Matrix
%                   S       Mode shapes
%                   Freq    Frequencies in rad/sec
%                   P       The eigenvectors of M^(-1/2)KM^(-1/2)


le=l/numel;
dof=numel*2+2;

m=rou*A*le/420*[ 156    22*le   54    -13*le;
                 22*le  4*le^2  13*le -3*le^2;
                 54     13*le   156   -22*le;
                -13*le -3*le^2 -22*le  4*le^2];

k=E*I/(le^3)*[ 12    6*le    -12    6*le;
               6*le  4*le^2  -6*le  2*le^2;
              -12   -6*le     12   -6*le;
               6*le  2*le^2  -6*le  4*le^2];
M = sparse(dof,dof); 
for i=1:2:dof-2
    point=[i i+1 i+2 i+3];
    M(point,point)=M(point,point)+m; 
end
K=sparse(dof,dof);
for i=1:2:dof-2
    point=[i i+1 i+2 i+3];
    K(point,point)=K(point,point)+k;
end

%============================================================
%============================================================
% Applying boundary conditions
% This part can be comented out for a free-free beam

M=M([3:end],[3:end]);
K=K([3:end],[3:end]);
%============================================================
%============================================================


%Li=chol(M);
%L=Li';
%Kt=inv(L)*K*inv(L');
%[P, fre2]=eig(Kt);
%Freq=sqrt(fre2);
%S=inv(L')*P;

%
%SI unit examples
%
%[M,K,S,Freq]=finbeam(.7112,2,2700,.00651*.02547,.00651^3*.02547/12,7.1e10)

% For the small beam
%[M,K,S,Freq]=finbeam(.6096,3,2700,1.22e-3*25.48e-3,1.22e-3^3*25.48e-3/12,7.1e10)

%[M,K,S,Freq]=finbeam(.4572,3,2700,1.22e-3*25.48e-3,1.22e-3^3*25.48e-3/12,7.1e10)

