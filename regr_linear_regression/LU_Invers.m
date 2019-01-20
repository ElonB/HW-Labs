function [ B ] = LU_Invers( A, L, U )
%LU_Invers 
%   Finding inverse of A by finding the inverse of lower(L)z = c
%   then solving with z for upper(U)b=x

sz = size(A);
n = sz(1);

C = eye(n);

% Forward substitution (Lz = c)
for i = 1 : n
    c = C(:,i);
    z=zeros(n,1);
    for j = 1 : n
        z(j) = c(j) / L(j,j);
        c(j+1 : n) = c(j+1 : n) - L(j+1 : n, j)*z(j);
    end
    Z(:, i) = z;
end

%Backward substitution (Ux = z)
for i = 1 : n
    z = Z(:, i);
    x = zeros(n,1);
    for j = n:-1:1
       x(j) = z(j) / U(j,j);
       z(1 : j-1) = z(1 : j-1) - U(1 : j-1, j)*x(j);
    end
    B(:, i) = x; %return value
end

end

