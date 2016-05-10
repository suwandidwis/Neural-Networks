%this is the artificial neural network testing program

%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%lh = length of hidden layer
%lx = length of input
%ly = length of output
% x = input matriks 
% t = target
% N = banyak data

benar=0;
salah=0;
tes=0;
RR=0;

%mengambil data untuk testing
filename= 'datadantarget1.xlsx';
sheetnya= 3;
xlrange='A3:C77';
sheett=2;
ranget='A3:C77';
target=xlsread(filename,sheett,ranget);

data_testing= xlsread(filename,sheetnya,xlrange);
x=zscore(data_testing);
N = length(x(:,1));
lx = length(x(1,:));
ly = length(target(1,:));

%nilai lx,lh,ly,N,v,v0,w,w0 dll sudah diinput dari backprop_modif
%feed forward
for n = 1:N
    %menghitung semua sinyal input dengan bobotnya
        for i = 1 : lh
            z_in(i) = v0(i) + x(n,:) * v(:,i);
            z(i) = 1/(1 + exp(-z_in(i)));
        end
        for j = 1 : ly
            y_in(j) = w0(j) + z * w(:,j);
            y(j) = 1/(1 + exp(-y_in(j)));
            
            %thresholding
            if y(j)> 0.8
                y(j)=1;
            else
                y(j)=0;
            end
        end
        
        %checking real data and output to get RR
        if target(n,:) == y
            benar=benar+1;
        else
            salah=salah+1;
        end

end

%checking datanya benar atau salah
tes=benar+salah;

if tes ~= N
    disp( 'banyak data yang di kelaskan tidak sama dengan N');
end

%calculating the recognition rate
RR= benar/N * 100;
disp('RRnya=');
disp(RR);