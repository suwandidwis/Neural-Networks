function[data_urut, target]=urutkan(x,y)

	nilaimin=min(y(2:1000));
	nilaimax=max(y(2:1000));
	for i = 2: 1000
		yt(i,:) = (2*(y(i)-nilaimin))./(nilaimax-nilaimin) -1;
	end
	xy(1,:)=[0 0 0 x(1) 0 0 0];
	xy(2,:)=[0 0 x(1) x(2) 0 0 yt(1)];
	xy(3,:)=[0 x(1) x(2) x(3) 0 yt(1) yt(2)];
	for i = 4 : 1000
		xy(i,:) = [x(i-3) x(i-2) x(i-1) x(i) yt(i-3) yt(i-2) yt(i-1)];
	end

data_urut = xy;
target = yt;
end