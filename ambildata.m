%this is a script to read excel data for neural network training

filename= 'datadantarget.xlsx';
range= 'A3:D152';
sheet=1;
sheett=2;
ranget='A3:C77';
datanya=xlsread('datadantarget.xlsx',sheet,range);
target=xlsread('datadantarget.xlsx',sheett,ranget);
%sekarang z scorenya

xdalam_matriks=zscore(datanya);

%sekarang x nya udah dalam matrik 75*4
%mau dijadiin vektor satu kebawah sehingga disusun jadi
%x= [x11;x12;x13;x14;x21;x22;.....xNN]
%x=[];
%for i=1:75
 %   x=[x;xdalam_matriks(i,:)'];
%end %ini udah di test bener koks

%sudah boleh dimasukkin ke backprop sebagai input

alpha=input('Masukkan koefisien pembelajaran=');
lh=input('Masukkan banyak hidden layer=');
