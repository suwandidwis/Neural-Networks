function [Y,xfine] = fineTuning(input,vInv1,v0Inv1,wInv1,w0Inv1,vID,v0ID,wID,w0ID,alpha,lh,momentum)

r = input;
N = length(r(:,1));
lx = 7;
ly = 1;
vInv2=vInv1;
v0Inv2=v0Inv1;
wInv2=wInv1;
w0Inv2=w0Inv1;
wjk = transpose(wInv2);

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
while MSSE > 0.0001
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
            xID(n,:)=[0 0 Y_Inv1(1) Y_Inv1(2) 0 0 Y_ID(1)];
        elseif n==3
            xID(n,:)=[0 Y_Inv1(1) Y_Inv1(2) Y_Inv1(3) 0 Y_ID(1) Y_ID(2)];
        else
            xID(n,:) = [Y_Inv1(n-3) Y_Inv1(n-2) Y_Inv1(n-1) Y_Inv1(n) Y_ID(n-3) Y_ID(n-2) Y_ID(n-1)];
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
        Y_ID(n,:)=yID;
        x_ID(n,:)=xID(n,:);

        if n==1
            xInv2(n,:)=[0 0 0 0 0 0 Y_ID(1)];
        elseif n==2
            xInv2(n,:)=[0 0 Y_Inv2(1) 0 0 Y_ID(1) Y_ID(2)];
        elseif n==3
            xInv2(n,:)=[0 Y_Inv2(1) Y_Inv2(2) 0 Y_ID(1) Y_ID(2) Y_ID(3)];
        else
            xInv2(n,:) = [Y_Inv2(n-3) Y_Inv2(n-2) Y_Inv2(n-1) Y_ID(n-3) Y_ID(n-2) Y_ID(n-1) Y_ID(n)];
        end
        x_Inv2(n,:)=xInv2(n,:);

    %menghitung semua sinyal input dengan bobotnya
        for i = 1 : lh
            z_inInv2(i) = v0Inv2(i) + xInv2(n,:) * vInv2(:,i);
            zInv2(i) = (1 - exp(-z_inInv2(i)))/(1 + exp(-z_inInv2(i)));
        end
        for j = 1 : ly
            y_inInv2(j) = w0Inv2(j) + zInv2 * wInv2(:,j);
            yInv2(j) = (1 - exp(-y_inInv2(j)))/(1 + exp(-y_inInv2(j)));
        end

        %Backpropagation of error dari bobot w
        for i = 1: lh
            for j = 1 : ly
                dk(j) = (yInv1(j)-yInv2(j))* (((1+yInv2(j))*(1-yInv2(j)))*0.5);      %menghitung informasi error :
                deltaw(i,j) = alpha * dk(j) * zInv2(i) + momentum * w1(i,j);          %menghitung besarnya koreksi bobot unit output  
            end
        end
        deltaw0 = alpha * dk + momentum * w00;                               %menghitung koreksi error bias unit output
        w1 = deltaw;
        w00 = deltaw0;
        d_in = dk * wjk;                                      %menghitung semua koreksi error
        for j = 1 : lx
            for i = 1 : lh
                dj(i) = d_in(i) * (((1+zInv2(i))*(1-zInv2(i)))*0.5);           %menghitung nilai aktivasi koreksi error
                deltav(j,i) = alpha * dj(i) * x_Inv2(n,j) + momentum * v1(j,i);          %menghitung koreksi bobot unit hidden  
            end    
        end
        deltav0 = alpha * dj + momentum * v00;                          %menghitung koreksi error bias unit hidden
        v1 = deltav;
        v00 = deltav0;

      %update bobot
        wInv1 = wInv1 + deltaw;
        w0Inv1 = w0Inv1 + deltaw0;
        vInv1 = vInv1 + deltav;
        v0Inv1 = v0Inv1 + deltav0;
        wInv2 = wInv2 + deltaw;
        w0Inv2 = w0Inv2 + deltaw0;
        vInv2 = vInv2 + deltav;
        v0Inv2 = v0Inv2 + deltav0;

        error(n)= 0.5*(Y_Inv1(n,:)-yInv2)*(Y_Inv1(n,:)-yInv2);
        Y_Inv2(n,:)=yInv2;

    end
    epoch = epoch+1;
    errortotal(epoch) = sum(error);
    clc
    epoch
    MSSE=sum(error)/N
    time=toc
    save('fineTuning.mat','errortotal','Y_Inv1','x_Inv1','Y_ID','x_ID','Y_Inv2','x_Inv2');
    save('bobotInvFineTuning.mat','wInv1','w0Inv1','vInv1','v0Inv1','wInv2','w0Inv2','vInv2','v0Inv2');
end
end