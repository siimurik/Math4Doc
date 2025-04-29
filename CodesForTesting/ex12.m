clc
clear

s = [0.0, 2.0,  4.0,  6.0];
t = [1.0, 9.0, 41.0, 41.0];
figure(1)
plot(s,t,'o')
xlabel('s')
ylabel('t')
grid on

ss = [s(1): 0.001: s(length(s))];
S3_2 = interp1(s, t, ss, "spline");
hold on
plot(ss, S3_2, 'r')
hold off

g0 = lagrange_poly([0.0, 2.0], [1.0, 9.0])

%-----------------------------------------------------------------
function lagrp = lagrange_poly(x, y)
    % Validate input lengths
    if length(x) ~= length(y)
        error('Input vectors x and y must have the same length.');
    end
    
    % Get the number of points
    n = length(x);
    
    % Define symbolic variable
    syms X;
    
    % Initialize the Lagrange polynomial
    lagrp_sym = 0;
    
    % Construct the Lagrange polynomial
    for i = 1:n
        % Compute the Lagrange basis polynomial L_i(X)
        Li = 1;
        for j = 1:n
            if i ~= j
                Li = Li * (X - x(j)) / (x(i) - x(j));
            end
        end
        
        % Add the contribution of the current basis polynomial to the total
        lagrp_sym = lagrp_sym + Li * y(i);
    end
    
    % Simplify the resulting polynomial
    lagrp_sym = expand(lagrp_sym);
    
    % Convert to a MATLAB function for numerical evaluation
    lagrp = matlabFunction(lagrp_sym);
end