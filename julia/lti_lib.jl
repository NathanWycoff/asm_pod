# Various functions usefule for Linear time-invariant systems.

function fourrier_basis(t)
    return(sin.(sigmas.*t));
end

function solve_lti(A, B, C, D, control)
    P = size(B)[2];
    N = size(A)[2];

    u0 = zeros(N);
end
