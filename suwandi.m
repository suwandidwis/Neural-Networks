function [finalerror,v,w,v0,w0,Y,epoch] = suwandi(x,alpha,t,lh,momentum)
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

errortotal = 1;
MSSE = errortotal/N;
error = 0;
epoch = 0;

while errortotal > 0.01
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
                deltaw(i,j) = alpha * dk(j) * z(i);          %menghitung besarnya koreksi bobot unit output  
            end
        end
        deltaw0 = alpha * dk;                               %menghitung koreksi error bias unit output
        d_in = dk * wjk;                                      %menghitung semua koreksi error
        for j = 1 : lx
            for i = 1 : lh
                dj(i) = d_in(i) * z(i)*(1-z(i));           %menghitung nilai aktivasi koreksi error
                deltav(j,i) = alpha * dj(i) * x(n,j);          %menghitung koreksi bobot unit hidden  
            end    
        end
        deltav0 = alpha * dj;                          %menghitung koreksi error bias unit hidden

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
    save('hasil_training.mat','MSSE','v','w','v0','w0','Y','epoch');
    
    if mod(epoch,100)==0
        figure(1)
        plot(errortotal)     
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
