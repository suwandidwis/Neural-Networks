function [finalerror,v,w,v0,w0,Y,epoch] = suwandi_v6(alpha,lh,momentum)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%lh = length of hidden layer
%lx = length of input
%ly = length of output
% x = input matriks 
% t = target
% N = banyak data

%Data preprocessing
filename= 'datadantarget.xlsx';
sheet=2;
sheett=2;
range= 'A1:D150';
ranget='A3:C152';
datanya=xlsread('datairis.xlsx',sheet,range);
t=xlsread(filename,sheett,ranget);
x=zscore(datanya);

lx = length(x(1,:));
N = 0.5*length(x(:,1));%diambil 50% data untuk proses pembelajaran
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
error = 0;
epoch = 0;

%Momentum
w1=zeros(lh,ly);
v1=zeros(lx,lh);
w00=zeros(1,ly);
v00=zeros(1,lh);

%Training
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
    errortotal(epoch)
    save('hasil_training.mat','errortotal','v','w','v0','w0','Y','epoch');
    if mod(epoch,200)==0
        figure(1)
        plot(errortotal)     
    end

%Testing
benar=0;
salah=0;
tes=0;
RR=0;

for n = 76 : 150
        %menghitung semua sinyal input dengan bobotnya
        for i = 1 : lh
            z_in(i) = v0(i) + x(n,:) * v(:,i);
            z(i) = 1/(1 + exp(-z_in(i)));
        end
        for j = 1 : ly
            y_in(j) = w0(j) + z * w(:,j);
            y(j) = 1/(1 + exp(-y_in(j)));
            
            %thresholding
            if y(j)> 0.6
                y(j)=1;
            else
                y(j)=0;
            end
        end
        YT(n-75,:)=y;
        %checking real data and output to get RR
        if t(n-75,:) == y
            benar=benar+1;
        else
            salah=salah+1;
        end
end
    %checking datanya benar atau salah
    tes=benar+salah;

    if tes ~= (150-N)
        disp( 'banyak data yang di kelaskan tidak sama dengan N');
    end

    %calculating the recognition rate
    RR= (benar/N) * 100;
    save('hasil_testing.mat','RR','benar','salah','tes','YT');

end
    finalerror=errortotal(epoch)
end
