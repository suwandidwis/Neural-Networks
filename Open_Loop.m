function[Y_ID]=Open_Loop

load('Input_NNID18.mat');
load('Hasil_NNID18.mat');
    vID=v;
    v0ID=v0;
    wID=w;
    w0ID=w0;
load('Hasil_NNINV14.mat');
    vInv1=v;
    v0Inv1=v0;
    wInv1=w;
    w0Inv1=w0;
r = t;
N = length(r(:,1));
lx = 7;
ly = 1;

%Open Loop System
	for n = 1:N
        if n==1
            xInv1(n,:)=[0 0 0 0 0 0 r(1)];
        elseif n==2
            xInv1(n,:)=[0 0 Y_Inv1(1) 0 0 r(1) r(2)];
        elseif n==3
            xInv1(n,:)=[0 Y_Inv1(1) Y_Inv1(2) 0 r(1) r(2) r(3)];
        else
            xInv1(n,:) = [Y_Inv1(n-3) Y_Inv1(n-2) Y_Inv1(n-1) r(n-3) r(n-2) r(n-1) r(n)];
        end

    %menghitung semua sinyal input dengan bobotnya
        for i = 1 : lh
            z_inInv1(i) = v0Inv1(i) + xInv1(n,:) * vInv1(:,i);
            zInv1(i) = (1 - exp(-z_inInv1(i)))/(1 + exp(-z_inInv1(i)));
        end
        for j = 1 : ly
            y_inInv1(j) = w0Inv1(j) + zInv1 * wInv1(:,j);
            yInv1(j) = (1 - exp(-y_inInv1(j)))/(1 + exp(-y_inInv1(j)));
        end
        Y_Inv1(n,:)=yInv1;
        x_Inv1(n,:)=xInv1(n,:);

        if n==1
            xID(n,:)=[0 0 0 Y_Inv1(1) 0 0 0];
        elseif n==2
            xID(n,:)=[0 0 Y_Inv1(1) Y_Inv1(2) 0 0 r(1)];
        elseif n==3
            xID(n,:)=[0 Y_Inv1(1) Y_Inv1(2) Y_Inv1(3) 0 r(1) r(2)];
        else
            xID(n,:) = [Y_Inv1(n-3) Y_Inv1(n-2) Y_Inv1(n-1) Y_Inv1(n) r(n-3) r(n-2) r(n-1)];
        end

    %menghitung semua sinyal input dengan bobotnya
        for i = 1 : lh
            z_inID(i) = v0ID(i) + xID(n,:) * vID(:,i);
            zID(i) = (1 - exp(-z_inID(i)))/(1 + exp(-z_inID(i)));
        end
        for j = 1 : ly
            y_inID(j) = w0ID(j) + zID * wID(:,j);
            yID(j) = (1 - exp(-y_inID(j)))/(1 + exp(-y_inID(j)));
        end
        error(n)= 0.5*(r(n,:)-yID)*(r(n,:)-yID);
        Y_ID(n,:)=yID;
        x_ID(n,:)=xID(n,:);
    end
    MSSE_DIC=sum(error)/N;
    save('NNDIC_Open15.mat','MSSE_DIC','Y_Inv1','x_Inv1','Y_ID','x_ID');
end