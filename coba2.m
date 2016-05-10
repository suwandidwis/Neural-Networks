function [Z,Y,v,w,v0,w0,DK,DELTAW,DELTAW0,DJ,DELTAV,DELTAV0] = coba2(input,target,alpha,lh)

x = input;
t = target;
N = 497;
lx = length(x(1,:));
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

%Training
    for n = 1:N
    %menghitung semua sinyal input dengan bobotnya
        for i = 1 : lh
            z_in(i) = v0(i) + x(n,:) * v(:,i);
            z(i) = (1 - exp(-z_in(i)))/(1 + exp(z_in(i)));
        end
        for j = 1 : ly
            y_in(j) = w0(j) + z * w(:,j);
            y(j) = (1 - exp(-y_in(j)))/(1 + exp(y_in(j)));
        end

    %Backpropagation of error dari bobot w
        for i = 1: lh
            for j = 1 : ly
                dk(j) = (t(n,j)-y(j))* (((1+y(j))*(1-y(j)))/2);      %menghitung informasi error :
                deltaw(i,j) = alpha * dk(j) * z(i);          %menghitung besarnya koreksi bobot unit output  
            end
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

Z=z;
Y=y;
DK=dk;
DELTAW=deltaw;
DELTAW0=deltaw0;
DJ=dj;
DELTAV=deltav;
DELTAV0=deltav0;
end