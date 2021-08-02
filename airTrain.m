function [dlnet, learnrate, totalTrainTime] = airTrain(numEpochs, miniBatchSize, dsTrain, dlnet)

mbq = minibatchqueue(dsTrain,...
    'MiniBatchSize',miniBatchSize,...
    'MiniBatchFcn', @airPreproMinibatch,...
    'MiniBatchFormat',{'SSCB', 'SSCB'});


% visualize training plot
plots = "training-progress";

if plots == "training-progress"
    figure
    lineLossTrain = animatedline('Color',[0.85 0.325 0.098]);
    ylim([0 inf])
    xlabel("Iteration")
    ylabel("Loss")
    grid on
end

% initialize ADAM
trailingAvg = [];
trailingAvgSq = [];

learnrate = 0.0006;
iteration = 0;
start = tic;

% Loop over epochs.
for epoch = 1:numEpochs
    
    % Shuffle data.
    shuffle(mbq)
    
    % Loop over mini-batches
    while hasdata(mbq)
        
        iteration = iteration + 1;
        
        [dlX,dlY] = next(mbq);
                       
        % Evaluate the model gradients, state, and loss using dlfeval and the
        % modelGradients function.
        [gradients,state,loss] = dlfeval(@airmodelGradients, dlnet, dlX, dlY);
        dlnet.State = state;
        
        % Update the network parameters using the Adam optimizer.
        [dlnet,trailingAvg,trailingAvgSq] = adamupdate(dlnet,gradients, ...
            trailingAvg,trailingAvgSq,iteration, learnrate);
        
        % Display the training progress.
        if plots == "training-progress"
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            addpoints(lineLossTrain,iteration,double(gather(extractdata(loss))))
            title("Epoch: " + epoch + ", Elapsed: " + string(D))
            drawnow
        end
    end
end
totalTrainTime = D;

end