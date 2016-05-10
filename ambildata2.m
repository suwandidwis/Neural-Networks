%this is a script to read excel data for neural network training

filename= 'datasbp.xlsx';
range= 'A3:D152';
sheet=1;
sheett=2;
sheetout=3;
ranget='A3:C77';
rangeout='A3:A77';
rangetest='A78:D152';
datanya=xlsread('datadantarget.xlsx',sheet,range);
target=xlsread('datadantarget.xlsx',sheett,ranget);
targetout=xlsread('datadantarget.xlsx',sheetout,rangeout);
datatest=xlsread('datadantarget.xlsx',sheet,rangetest);
%sekarang z scorenya

xdalam_matriks=zscore(datanya);
xtest=zscore(datatest);

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
