function wavefield_3d()

    vel = load('vel.mat').res;
    u = load('test9.mat').res;
    [xx,yy] = meshgrid(0:1:256);
    plt = .5 * interp2(u,xx,yy) + .05;
 	surf(plt,'FaceColor',[0.5 0.5 .5]);
    
    hold on;
    contour(vel)
    zt = get(gca, 'ZTick');
    set(gca, 'ZTick',zt, 'ZTickLabel',zt - .05)
    
    set(gca,'XTick',[])
    set(gca,'YTick',[])
    set(gca,'ZTick',[])
    
    xlabel('x_2')
    ylabel('x_1')
    zlabel('wave u')
    
end




