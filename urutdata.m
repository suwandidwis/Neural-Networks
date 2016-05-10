data = zeros(150,4);
setosa=xlsread('datairis.xlsx',2,'F1:I50');
virginica=xlsread('datairis.xlsx',2,'F51:I100');
versicolor=xlsread('datairis.xlsx',2,'F101:I150');
for i = 1 : 50
	data(3*i-2,:)=setosa(i,:);
	data(3*i-1,:)=virginica(i,:);
	data(3*i,:)=versicolor(i,:);
end
