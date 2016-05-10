function[data_urutppr, target]=urutkanppr(x,y)

	nilaimin_x=min (x);
    nilaimax_x=max (x);
	nilaimin_y=min (y);
    nilaimax_y=max (y);
	for i = 1: 5000
		yt(i,:) = (2*(y(i)-nilaimin_y))./(nilaimax_y-nilaimin_y) -1;
        xt(i,:) = (2*(x(i)-nilaimin_x))./(nilaimax_x-nilaimin_x) -1;
    end

    xy(1,:)=[0 0 0 xt(1) 0 0 0];
	xy(2,:)=[0 0 xt(1) xt(2) 0 0 yt(1)];
	xy(3,:)=[0 xt(1) xt(2) xt(3) 0 yt(1) yt(2)];

	for i = 4 : 5000
		xy(i,:) = [xt(i-3) xt(i-2) xt(i-1) xt(i) yt(i-3) yt(i-2) yt(i-1)];
	end

data_urutppr = xy;
target = yt;
end