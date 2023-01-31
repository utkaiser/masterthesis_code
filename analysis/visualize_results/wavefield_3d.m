function wavefield_3d()
    
    tensor = load('diagonal_fine.mat');
    vel = load('diagonal.mat');
    vel = vel.res;
    v = 1;
    [xx, yy] = meshgrid(0:v:128);
    Z = squeeze(tensor.res(:,:));
    plt = interp2(Z,xx,yy);
	surf(plt);
    
end