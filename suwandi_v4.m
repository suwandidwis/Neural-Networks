function [finalerror,v,w,v0,w0,Y,epoch,YT,out,rec_rate] = suwandi_v4(x,alpha,t,lh,momentum,tout)
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
beta = 0.7*lh^(1/lx);
v = rand(lx,lh)-0.5*ones(lx,lh);
w = rand(lh,ly)-0.5*ones(lh,ly);
v0 = -beta + (beta+beta)*rand(1,lh);
w0 = -beta + (beta+beta)*rand(1,ly);
norm_v = zeros(1,lh);
norm_w = zeros(1,ly);
for j = 1 : lh
    for i = 1 : lx
        norm_v(j) = norm_v(j) + v(i,j)^2;
        v(i,j) = (beta/sqrt(norm_v(j)))*v(i,j);
    end
end

for j = 1 : ly
    for i = 1 : lh
        norm_w(j) = norm_w(j) + w(i,j)^2;
        w(i,j) = (beta/sqrt(norm_w(j)))*w(i,j);
    end
end
wjk = transpose(w);

errortotal = 100;
MSSE = errortotal/N;
error = 0;
epoch = 0;
rec_rate = 0;

%momentum
w1=zeros(lh,ly);
v1=zeros(lx,lh);
w00=zeros(1,ly);
v00=zeros(1,lh);

while MSSE > 0.01
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

% Testing
benar = 0;
salah = 0;
YT = zeros(75,3);

for n = 76 : 150
    for i = 1 : lh
        z_int(i) = v0(i) + x(n,:) * v(:,i);
        zt(i) = 1/(1 + exp(-z_int(i)));
    end
    for j = 1 : ly
        y_int(j) = w0(j) + zt * w(:,j);
        yt(j) = 1/(1 + exp(-y_int(j)));
    end
    YT(n-N,:) = yt;
    errortest(1) = 0.5*(t(1,:)-yt)*(t(1,:)-yt)';
    errortest(2) = 0.5*(t(2,:)-yt)*(t(2,:)-yt)';
    errortest(3) = 0.5*(t(3,:)-yt)*(t(3,:)-yt)';
    if (errortest(1) < errortest(2)) && (errortest(1) < errortest(3))
        out(n-N) = 1;
    else if (errortest(2) < errortest(1)) && (errortest(2) < errortest(3))
        out(n-N) = 2;
    else if (errortest(3) < errortest(1)) && (errortest(3) < errortest(2));
        out(n-N) = 3;
    end
end

for n = 1 : N
    if (out(n) == tout(n))
        benar = benar + 1;
    else
        salah = salah + 1;
    end
end
    rec_rate = (benar/(150-N))*100;

end
    finalerror=errortotal(epoch)
    v
    w
    v0
    w0
    Y
    epoch
    YT
    out
    rec_rate
    benar
    salah
end
