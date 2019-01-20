%Regularized linear model (polynomial basis) regression
% Homework 1, 5255 Machine Learning (5255)
%Elon Brange

clear;
clc;

%Menu
menu = questdlg('Choose', ...
                'Menu', ...
                'Input','Standard test','Standard test');        

%File path 1: dat file
% 1 year BitCoin weighted closing value

%Handle menu
switch menu
    case 'Input'
        input = {'Enter fie name (FILENAME.dat)',
            'Enter no. of pol. bases(n)',
            'Enter value for lambda'};
        res = inputdlg(input,'Input data');
        
        %data = xlsread(string(res(1)));
        data = csvread(string(res(1)));
        phi = str2num(cell2mat(res(2)));
        lambda = cell2mat(res(3));
        
    case 'Standard test'
        %data = xlsread('Data.xlsx');
        data = csvread('BCData2017.dat');
        phi = 10;
        lambda = 5;
end

x = data(:,1);
y =data(:,2);
plot(x,y,'o'); 
hold;

I = eye(phi);
b = y;

%Design matrix A
A = (ones(size(x)));
for i = 1 : phi-1
    t = x.^i;
    A = [t A];
end


%LU decomposition functions
temp = A'*A - lambda*I;
[L U] = LU_Decomp(temp);
B = LU_Invers(temp, L, U);

%Reg function (MLE)
xhat = B*A'*b;

%Plotting
v = linspace(min(x), max(x));
p = xhat';

for i = 1 : length(v)
    f(i) = polyval(p,v(i));
end

%Printing equation
P = poly2sym(xhat);
t = string(P)

%Plotting x, y and fitted regularized model
title('Regularized linear model (polynomial basis: phi=' + string(phi) +' lambda=' + string(lambda));
legend('Data', 'Location','southeast');
plot(v,f);
%text(5,5,t);
annotation('textbox', [0, 0.5, 0, 0], 'string', t);



