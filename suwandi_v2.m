function [finalerror,v,w,v0,w0,Y,epoch] = suwandi_v2(x,alpha,t,lh,momentum)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%lh = length of hidden layer
%lx = length of input
%ly = length of output
% x = input matriks 
% t = target
% N = banyak data

lx = length(x(1,:));
N = 0.5*length(x(:,1));%diambil 50% data untuk proses pembelajaran
ly = length(t(1,:));

%inisialisasi nilai bobot
v = rand(lx,lh);
w = rand(lh,ly);
v0 = rand(1,lh);
w0 = rand(1,ly);
wjk = transpose(w);

errortotal = 100;
MSSE = errortotal/N;
error = 0;
epoch = 0;

%momentum
w1=zeros(lh,ly);
v1=zeros(lx,lh);
w00=zeros(1,ly);
v00=zeros(1,lh);

while MSSE > 0.00001
    for n = 1:N
    %menghitung semua sinyal input dengan bobotnya
        for i = 1 : lh
            z_in(i) = v0(i) + x(n,:) * v(:,i);
            z(i) = 1/(1 + exp(-z_in(i)));
        end
        for j = 1 : ly
            y_in(j) = w0(j) + z * w(:,j);
            y(j) = 1/(1 + exp(-y_in(j)));
        end

    %Backpropagation of error dari bobot w
        for i = 1: lh
            for j = 1 : ly
                dk(j) = (t(n,j)-y(j))* y(j)*(1-y(j));      %menghitung informasi error :
                deltaw(i,j) = alpha * dk(j) * z(i) + momentum * w1(i,j);          %menghitung besarnya koreksi bobot unit output  
            end
        end
        deltaw0 = alpha * dk + momentum * w00;                               %menghitung koreksi error bias unit output
        w1 = deltaw;
        w00 = deltaw0;
        d_in = dk * wjk;                                      %menghitung semua koreksi error
        for j = 1 : lx
            for i = 1 : lh
                dj(i) = d_in(i) * z(i)*(1-z(i));           %menghitung nilai aktivasi koreksi error
                deltav(j,i) = alpha * dj(i) * x(n,j) + momentum * v1(j,i);          %menghitung koreksi bobot unit hidden  
            end    
        end
        deltav0 = alpha * dj + momentum * v00;                          %menghitung koreksi error bias unit hidden
        v1 = deltav;
        v00 = deltav0;

      %update bobot
        w = w + deltaw;
        w0 = w0 + deltaw0;
        v = v + deltav;
        v0 = v0 + deltav0;

        error(n)= 0.5*(t(n,:)-y)*(t(n,:)-y)';
        Y(n,:)=y;
    end
    epoch = epoch+1;
    errortotal(epoch) = sum(error);
    clc
    epoch
    MSSE=sum(error)/75
    %figure(1)
    %plot(errortotal)
    
    if mod(epoch,100)==0
        figure(1)
        plot(errortotal/75)     
    end       
end
    finalerror=errortotal(epoch)
    v
    w
    v0
    w0
    Y
    epoch
end
