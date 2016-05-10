function [YTest, MSSETest] = feedforwardNNID

load('Input_NNIDPPR03.mat');
load('Hasil_NNIDPPR03.mat');
N = 0.5*length(x(:,1));
lx = length(x(1,:));
ly= length(t(1,:));

	for n = (N+1) : (2*N)
    %menghitung semua sinyal input dengan bobotnya
        for i = 1 : lh
            z_in(i) = v0(i) + x(n,:) * v(:,i);
            z(i) = (1 - exp(-z_in(i)))/(1 + exp(-z_in(i)));
        end
        for j = 1 : ly
            y_in(j) = w0(j) + z * w(:,j);
            y(j) = (1 - exp(-y_in(j)))/(1 + exp(-y_in(j)));
        end

        error(n-N)= 0.5*(t(n,:)-y)*(t(n,:)-y);
    	YTest(n-N,:)=y;
    end
    MSSETest=sum(error)/N;
    save('Hasil_TestingNNIDPPR03.mat','YTest','MSSETest');
    
    %Plot MSSE
    figure(1)
    plot(errortotal/N)
    title('Grafik MSSE')
    
    %Plot data training
    figure(2)
    plot(t(1:N),'r')
    hold on
    plot(Y,'b')
    title('Data Training')
    legend('Plant Output','NN ID Output')
    hold off
    
    %Plot data testing
    figure(3)
    plot(t(N+1:2*N),'r')
    hold on
    plot(YTest,'b')
    title('Data Testing')
    legend('Plant Output','NN ID Output')
    hold off
end
