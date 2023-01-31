function pca_wavefield_dataset()
    
    tensor = load('results/datagen/pca3_scaled.mat');
    tensor = squeeze(tensor.res);
    x = tensor(:,1);
    y = tensor(:,2);
    z = tensor(:,3);
    
    
%     c = {[1 1 1]; [0 1 0]; [0 0 1]; [0 0 0]; [1 1 0]; [1 0 1]; [0 1 1]; [1 0 0]; [0.4660 0.6740 0.1880]; [0.9290 0.6940 0.1250]}; 
%     for i=0:10:590
%         for j=1:10
%             if j ~= 1
%                 a = cell2mat(c(j));
%                 scatter3(x(i+j:i+j),y(i+j:i+j),z(i+j:i+j),30,a, 'filled');
%                 hold on;
%             end
%             
%         end
%     end
%     legend([{'1'},{'2'},{'3'}, {'4'}, {'5'}, {'6'}, {'7'},{'8'},{'9'}])
%     savefig('results/datagen/pca3_dataset_snapshots.fig')
     
     
    c = {[1 0 0]; [0 1 0]; [0 0 1]; [0 0 0]; [1 1 0]; [1 0 1]};
    prev = 1;
    counter = 1;
    array = [];
    for i=[100,200,300,400,500,600]
        
        a = cell2mat(c(counter));
        axis = scatter3(x(prev:i),y(prev:i),z(prev:i),30,a,'filled');
        array = [array,axis];
        hold on;
        
        for j=0:10:90
            xPair = [x(prev+j:prev+9+j), x(prev+j:prev+9+j)];
            yPair = [y(prev+j:prev+9+j), y(prev+j:prev+9+j)];
            zPair = [z(prev+j:prev+9+j), z(prev+j:prev+9+j)];

            plot3(xPair,yPair,zPair, 'Color', a);
            hold on;            
        end
        
        prev = i + 1;
        counter = counter + 1;
    end
%     legend([p1 p3],{'First','Third'})
    legend(array, {'diagonal', 'three layers', 'cracks', 'high frequency', 'BP', 'Marmousi'})
%     savefig('results/datagen/pca3_dataset_velocities.fig')
    
    
end