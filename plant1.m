function[input,output] = plant1

x= 2*rand(1001,1) - 1;
y(1)=0;
	for i = 2 : 1001
		y(i,1) = (1/(1+y(i-1)^2) - 0.25*x(i) - 0.3*x(i-1));
	end

input = x(2:1001);
output = y(2:1001);
save('dataPlant3.mat','input','output');
end
