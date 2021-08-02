train_data_all = zeros(128,128,6,100);


for i = 1:64
    path = '/Users/rnbrown/Documents/MATLAB/Airfoil_Project/Airfoil_Training-mat-files/train_data_chunk%d.mat';
    idx = i;
    currentFile = sprintf(path, idx);
    load(currentFile);
    train_data_all = cat(4, train_data_all, train_data_full);
    
    
   
    
end