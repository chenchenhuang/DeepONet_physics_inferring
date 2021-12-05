
% MATLAB ODE45 %

% SOLVE  du/dt = a1 d2u/dx2 - a2 u^3 + a3 u
% initial condition: u(t=0) = x^2 cos(pi x)
% boundary condition: u(x=1) = u(x=-1), du/dx(x=1) = du/dx(x=-1) 
function [u, x, t] = AllenEQ(a1,a2,a3,N)
% N = 1000;
t = linspace(0, 1, N);
x = linspace(-1, 1, N)';
dx = x(2)-x(1);

u0(:,1) = x.^2.*cos(pi*x);

u = zeros(length(x), length(t));
f = @(t,u) RHS(u, dx, N, a1, a2, a3);
[t, u]=ode45( f, t, u0 );

function dudt = RHS( u, dx, N, a1, a2, a3)
    
    u = u(:);
    for j = 2:N-1
        uxx(j) = ( u(j+1) - 2*u(j) + u(j-1) )/dx^2;
    end
    uxx(1) = ( u(2) - 2*u(1) + u(N) )/dx^2;
    uxx(N) = ( u(1) - 2*u(N) + u(N-1) )/dx^2;
%     uxx(1) = ( 2*u(N)-u(1) - u(N-1) )/dx^2;
%     uxx(N) = ( 2*u(1) - u(N) - u(2) )/dx^2;
    uxx = uxx(:);
    
    dudt = a1*uxx -a2*u.^3 + a3*u;
end

end
