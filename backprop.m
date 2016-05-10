function [finalerror,v,w,v0,w0,Y,epoch] = backprop(alpha,lh)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%lh = length of hidden layer
%lx = length of input
%ly = length of output
% x = input matriks 
% t = target
% N = banyak data

data = 'mydata.txt';
[A,delimiterOut]=importdata(data);
t = eye(3);

lx = length(A(1,:));
N = length(A(:,1));			%diambil 50% data untuk proses pembelajaran
ly = length(t(1,:));

%inisialisasi nilai bobot
v = rand(lx,lh);
w = rand(lh,ly);
v0 = rand(1,lh);
w0 = rand(1,ly);

w1=zeros(lh,ly);
v1=zeros(lx,lh);
w00=zeros(1,ly);
v00=zeros(1,lh);

errortotal = 100;
error = 0;
epoch = 0;

while errortotal > 0.01
    for n = 1:N
    %menghitung semua sinyal input dengan bobotnya
        for i = 1 : lh
            z_in(i) = v0(i) + A(n,:) * v(:,i);
            z(i) = 1/(1 + exp(-z_in(i)));
        end
        for j = 1 : ly
            y_in(j) = w0(j) + z * w(:,j);
            y(j) = 1/(1 + exp(-y_in(j)));
        end

    %Backpropagation of error dari bobot w
        for j = 1 : ly
            d(j) = (t(n,j)-y(j))* y(j)*(1-y(j));      %menghitung informasi error :
            deltaW(:,j) = alpha * d(j) * z(j);          %menghitung besarnya koreksi bobot unit output 
        end

        deltaw0 = alpha * d;                          %menghitung besarnya koreksi bias output
        w1 = deltaW;
        w00 = deltaW0;
        for i = 1 : lh
            d_in(i) = d(j) * w(j);                    %menghitung semua koreksi error
            d(i) = d_in(i) * z(j)*(1-z(j));           %menghitung nilai aktivasi koreksi error
            deltaV(:,i) = alpha * d(i) * A(i);          %menghitung koreksi bobot unit hidden
        end
        deltav0 = alpha * d;                          %menghitung koreksi error bias unit hidden
        v1 = deltaV;
        v00 = deltaV0;

    %update bobot  
        w = w + deltaW;
        w0 = w0 + deltaW0;
        v = v + deltaV;
        v0 = v0 + deltaV0;

        error(n)= 0.5*(t(n,:)-y)*(t(n,:)-y)';
        Y(n,:)=y;

    end
    epoch = epoch+1;
    errortotal(epoch) = sum(error);
    clc
    epoch
    errortotal(epoch)
    figure(1)
    plot(errortotal)
end
    finalerror=errortotal(epoch);
end