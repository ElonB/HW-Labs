function [ L, U ] = LU_Decomp( A )
%LU Composition function

%Square matrix test, Thows error if not nxn matrix
 sz = size(A);
 if sz(1) ~= sz(2)
     print('ERROR: A is not nxn matrix, LU decomp. not applicable');
     return;
 end
 
 sz = size(A);
 n =sz(1);
 
 L = zeros(n);
 U = zeros(n);
 
 for i = 1 : n
     %L
     for k = 1: i-1
        L(i,k) = A(i, k);
        for j = 1 : k - 1
            L(i, k) = L(i, k) - L(i, j)*U(j,k);            
        end
        L(i, k) = L(i, k) / U(k, k);
     end
     
     % U
     for k = i : n
         U(i, k) = A(i, k);
         for j = 1 : i-1
             U(i, k) = U(i, k) - L(i, j) * U(j, k);
         end          
     end   
 end
 
 for i = 1 : n
     L(i, i) = 1;
 end

end

