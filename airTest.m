function [YPred, diff] = airTest(miniBatchSize, dsTest, dlnet)

mbqTest = minibatchqueue(dsTest,...
    'MiniBatchSize',miniBatchSize,...
    'MiniBatchFcn', @airPreproMinibatch,...
    'MiniBatchFormat',{'SSCB','SSCB'});

YPred = [];
diff = [];

% Loop over mini-batches.
while hasdata(mbqTest)
    
    % Read mini-batch of data.
    [dlXTest,dlYTest] = next(mbqTest);
    
    % Make predictions using the predict function.
    [dlYPred] = predict(dlnet,dlXTest,'Outputs',["resize-scale_7"]);
    
    % Dermine predicted angles
    YPredBatch = extractdata(dlYPred);   
    YPred = cat(4, YPred, YPredBatch);
    
    % Compare predicted and true angles
    diffBatch = YPredBatch - dlYTest;
    diff = cat(4, diff, extractdata(gather(diffBatch)));
    
end

end