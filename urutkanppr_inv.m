function[data_urut, target]=urutkanppr_inv(x)

	nilaimin_x=min(x(:,2));
	nilaimax_x=max(x(:,2));
	nilaimin_y=min(x(:,3));
	nilaimax_y=max(x(:,3));
	for i = 1: 5000
		x_norm(i,:) = (2*(x(i,2)-nilaimin_x))./(nilaimax_x-nilaimin_x) -1;
		y_norm(i,:) = (2*(x(i,3)-nilaimin_y))./(nilaimax_y-nilaimin_y) -1;
	end
	xy(1,:)=[0 0 0 0 0 0 y_norm(1)];
	xy(2,:)=[0 0 x_norm(1) 0 0 y_norm(1) y_norm(2)];
	xy(3,:)=[0 x_norm(1) x_norm(2) 0 y_norm(1) y_norm(2) y_norm(3)];
	for i = 4 : 5000
		xy(i,:) = [x_norm(i-3) x_norm(i-2) x_norm(i-1) y_norm(i-3) y_norm(i-2) y_norm(i-1) y_norm(i)];
	end

data_urut = xy;
target = x_norm;
end