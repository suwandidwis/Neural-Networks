function [finalerror,v,w,v0,w0,Y,epoch] = backpropagationINVPPR(io,alpha,lh,momentum)

%Pengolahan data input
nilaimin_x=min(io(:,2));
nilaimax_x=max(io(:,2));
nilaimin_y=min(io(:,3));
nilaimax_y=max(io(:,3));
for i = 1: 5000
    target(i,:) = (2*(io(i,3)-nilaimin_y))./(nilaimax_y-nilaimin_y) -1;
    u_k(i,:) = (2*(io(i,2)-nilaimin_x))./(nilaimax_x-nilaimin_x) -1;
end
input(1,:)=[0 0 0 0 0 0 0 0 target(1)];
input(2,:)=[0 0 0 u_k(1) 0 0 0 target(1) target(2)];
input(3,:)=[0 0 u_k(1) u_k(2) 0  0 target(1) target(2) target(3)];
input(4,:)=[0 u_k(1) u_k(2) u_k(3) 0 target(1) target(2) target(3) target(4)];
for i = 5 : 5000
    input(i,:) = [u_k(i-4) u_k(i-3) u_k(i-2) u_k(i-1) target(i-4) target(i-3) target(i-2) target(i-1) target(i)];
end

%Inisialisasi data awal
x = input;
t = u_k;
N = 0.5*length(x(:,1));
lx = length(x(1,:));
ly = length(t(1,:));

%Inisialisasi nilai bobot
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

%Momentum
w1=zeros(lh,ly);
v1=zeros(lx,lh);
w00=zeros(1,ly);
v00=zeros(1,lh);

tic

%Training
while MSSE > 0.00001
    for n = 1:N
    %menghitung semua sinyal input dengan bobotnya
        for i = 1 : lh
            z_in(i) = v0(i) + x(n,:) * v(:,i);
            z(i) = (1 - exp(-z_in(i)))/(1 + exp(-z_in(i)));
        end
        for j = 1 : ly
            y_in(j) = w0(j) + z * w(:,j);
            y(j) = (1 - exp(-y_in(j)))/(1 + exp(-y_in(j)));
        end

    %Backpropagation of error dari bobot w
        for i = 1: lh
            for j = 1 : ly
                dk(j) = (t(n,j)-y(j))* (((1+y(j))*(1-y(j)))*0.5);      %menghitung informasi error :
                deltaw(i,j) = alpha * dk(j) * z(i) + momentum * w1(i,j);          %menghitung besarnya koreksi bobot unit output  
            end
        end
        deltaw0 = alpha * dk + momentum * w00;                               %menghitung koreksi error bias unit output
        w1 = deltaw;
        w00 = deltaw0;
        d_in = dk * wjk;                                      %menghitung semua koreksi error
        for j = 1 : lx
            for i = 1 : lh
                dj(i) = d_in(i) * (((1+z(i))*(1-z(i)))*0.5);           %menghitung nilai aktivasi koreksi error
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

        error(n)= 0.5*(t(n,:)-y)*(t(n,:)-y);
        Y(n,:)=y;
    end
    epoch = epoch+1;
    errortotal(epoch) = sum(error);
    clc
    epoch
    MSSE=sum(error)/N
    time=toc
    save('Input_NNINVPPR03.mat','x','t','alpha','lh','momentum');
    save('Hasil_NNINVPPR03.mat','MSSE','errortotal','v','w','v0','w0','v1','w1','v00','w00','Y','epoch','time');
end
    finalerror=errortotal(epoch)/N;
end