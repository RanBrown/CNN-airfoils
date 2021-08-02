function [gradients, state, loss] = airmodelGradients(dlnet, dlX, target)
            % the predictions Y and the training targets T.
            [dlY, state] = forward(dlnet,dlX,'Outputs',["resize-scale_7"]);
            % Define dimensions
            size(dlY)
            %A = size(out_p);
           
            % Calculate errors.
            loss_p = abs(dlY(:,:,1,:)-target(:,:,1,:));
            loss_u = abs(dlY(:,:,2,:)-target(:,:,2,:));
            loss_v = abs(dlY(:,:,3,:)-target(:,:,3,:));
            totLoss = (loss_p + loss_u + loss_v);
            loss = mean(totLoss,'all');
            
            gradients = dlgradient(loss,dlnet.Learnables);
            
            
end