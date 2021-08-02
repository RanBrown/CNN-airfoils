function [X,Y] = airPreproMinibatch(XCell,Ycell)
    
    % Extract image data from cell and concatenate
    X = cat(4,XCell{:});
    
    % Extract angle data from cell and concatenate
    Y = cat(4,Ycell{:});
    
    
end