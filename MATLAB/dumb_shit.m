function ret = dumb_shit(x)
    N = length(x) / 2;
    %A = diag(-exp(x(1:N)));
    A = diag(-(1:N));
    B = x(1:N);
    C = x((N+1):(2*N))';
    D = 0;

    sys = ss(A,B,C,D);
    svs = hsvd(sys);

    ret = log(svs(1)) - log(svs(end));
end
