function[y_NNID]=denormalisasi(y,y_k)

	nilaimin=min(y(2:1000));
	nilaimax=max(y(2:1000));
	for i = 2: 500
		y_NNID(i,:) = ((y_k(i)+1)*(nilaimax-nilaimin)*0.5) + nilaimin;
	end
end