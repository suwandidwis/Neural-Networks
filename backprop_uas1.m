function [finalerror,v,w,v0,w0,Y,epoch] = backprop_uas1(input,target,alpha,lh,momentum)

x = input;
t = target;
N = 498;
lx = 6;
ly = 1;

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

    for i = 1 : lh
        norm_w = norm_w + w(i,1)^2;
        w(i,1) = (beta/sqrt(norm_w)*w(i,1));
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

%Training
while MSSE > 0.01
    for n = 1:N
    %menghitung semua sinyal input dengan bobotnya
        for i = 1 : lh
            z_in(i) = v0(i) + x(n,:) * v(:,i);
            z(i) = (1 - exp(-z_in(i)))/(1 + exp(z_in(i)));
        end
            y_in = w0 + z * w(:,1);
            y = (1 - exp(-y_in))/(1 + exp(y_in));

        for i = 1: lh
                dk = (t(n+3,1)-y)* ((1+y)*(1-y)/2);      %menghitung informasi error :
                deltaw(i,1) = alpha * dk * z(i);          %menghitung besarnya koreksi bobot unit output  
        end
        deltaw0 = alpha * dk;                               %menghitung koreksi error bias unit output
        d_in = dk * wjk;                                      %menghitung semua koreksi error
        for j = 1 : lx
            for i = 1 : lh
                dj(i) = d_in(i) * (((1+z(i))*(1-z(i)))/2);           %menghitung nilai aktivasi koreksi error
                deltav(j,i) = alpha * dj(i) * x(n,j);          %menghitung koreksi bobot unit hidden  
            end    
        end
        deltav0 = alpha * dj;                          %menghitung koreksi error bias unit hidden

      %update bobot
        w = w + deltaw;
        w0 = w0 + deltaw0;
        v = v + deltav;
        v0 = v0 + deltav0;

        error(n)= 0.5*(t(n+3,:)-y)*(t(n+3,:)-y);
        Y(n,:)=y;
    end
    epoch = epoch+1;
    errortotal(epoch) = sum(error);
    clc
    epoch
    MSSE=sum(error)/N
    save('hasil_training2.mat','errortotal','v','w','v0','w0','Y','epoch');
    if mod(epoch,200)==0
        figure(1)
        plot(errortotal/N)     
    end

end
    finalerror=errortotal(epoch)
end