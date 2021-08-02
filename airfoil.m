function [dlnet] = airfoil()
airfoilNet_1 = layerGraph();

tempLayers = [
    imageInputLayer([128 128 3],"Name","imageinput", "Normalization", "none")
    convolution2dLayer([4 4],64,"Name","l_1","Padding","same","Stride",[2 2])
    dropoutLayer(0.01,"Name","dropout_1")];
airfoilNet_1 = addLayers(airfoilNet_1,tempLayers);

tempLayers = [
    leakyReluLayer(0.2,"Name","leakyrelu_1")
    convolution2dLayer([4 4],128,"Name","l_2","Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm_1")
    dropoutLayer(0.01,"Name","dropout_2")];
airfoilNet_1 = addLayers(airfoilNet_1,tempLayers);

tempLayers = [
    leakyReluLayer(0.2,"Name","leakyrelu_2")
    convolution2dLayer([4 4],128,"Name","l_3","Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm_2")
    dropoutLayer(0.01,"Name","dropout_3")];
airfoilNet_1 = addLayers(airfoilNet_1,tempLayers);

tempLayers = [
    leakyReluLayer(0.2,"Name","leakyrelu_3")
    convolution2dLayer([4 4],256,"Name","l_4","Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm_3")
    dropoutLayer(0.01,"Name","dropout_4")];
airfoilNet_1 = addLayers(airfoilNet_1,tempLayers);

tempLayers = [
    leakyReluLayer(0.2,"Name","leakyrelu_4")
    convolution2dLayer([2 2],512,"Name","l_5","Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm_4")
    dropoutLayer(0.01,"Name","dropout_5")];
airfoilNet_1 = addLayers(airfoilNet_1,tempLayers);

tempLayers = [
    leakyReluLayer(0.2,"Name","leakyrelu_5")
    convolution2dLayer([2 2],512,"Name","l_6","Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm_5")
    dropoutLayer(0.01,"Name","dropout_6")];
airfoilNet_1 = addLayers(airfoilNet_1,tempLayers);

tempLayers = [
    leakyReluLayer(0.2,"Name","leakyrelu_6")
    convolution2dLayer([2 2],512,"Name","l_7","Padding","same","Stride",[2 2])
    dropoutLayer(0.01,"Name","dropout_7")
    resize2dLayer("Name","resize-scale_1","GeometricTransformMode","half-pixel","Method","bilinear","NearestRoundingMode","round","Scale",[2 2])
    reluLayer("Name","relu_1")
    convolution2dLayer([1 1],512,"Name","l_8","Padding","same")
    batchNormalizationLayer("Name","batchnorm_6")];
airfoilNet_1 = addLayers(airfoilNet_1,tempLayers);

tempLayers = [
    concatenationLayer(3,2,"Name","concat_1")
    reluLayer("Name","relu_2")
    convolution2dLayer([1 1],512,"Name","l_9","Padding","same")
    batchNormalizationLayer("Name","batchnorm_7")
    resize2dLayer("Name","resize-scale_2","GeometricTransformMode","half-pixel","Method","bilinear","NearestRoundingMode","round","Scale",[2 2])];
airfoilNet_1 = addLayers(airfoilNet_1,tempLayers);

tempLayers = [
    concatenationLayer(3,2,"Name","concat_2")
    reluLayer("Name","relu_3")
    convolution2dLayer([3 3],256,"Name","l_10","Padding","same")
    batchNormalizationLayer("Name","batchnorm_8")
    resize2dLayer("Name","resize-scale_3","GeometricTransformMode","half-pixel","Method","bilinear","NearestRoundingMode","round","Scale",[2 2])];
airfoilNet_1 = addLayers(airfoilNet_1,tempLayers);

tempLayers = [
    concatenationLayer(3,2,"Name","concat_3")
    reluLayer("Name","relu_4")
    convolution2dLayer([3 3],128,"Name","l_11","Padding","same")
    batchNormalizationLayer("Name","batchnorm_9")
    resize2dLayer("Name","resize-scale_4","GeometricTransformMode","half-pixel","Method","bilinear","NearestRoundingMode","round","Scale",[2 2])];
airfoilNet_1 = addLayers(airfoilNet_1,tempLayers);

tempLayers = [
    concatenationLayer(3,2,"Name","concat_4")
    reluLayer("Name","relu_5")
    convolution2dLayer([3 3],128,"Name","l_12","Padding","same")
    batchNormalizationLayer("Name","batchnorm_10")
    resize2dLayer("Name","resize-scale_5","GeometricTransformMode","half-pixel","Method","bilinear","NearestRoundingMode","round","Scale",[2 2])];
airfoilNet_1 = addLayers(airfoilNet_1,tempLayers);

tempLayers = [
    concatenationLayer(3,2,"Name","concat_5")
    reluLayer("Name","relu_6")
    convolution2dLayer([3 3],64,"Name","l_13","Padding","same")
    batchNormalizationLayer("Name","batchnorm_11")
    resize2dLayer("Name","resize-scale_6","GeometricTransformMode","half-pixel","Method","bilinear","NearestRoundingMode","round","Scale",[2 2])];
airfoilNet_1 = addLayers(airfoilNet_1,tempLayers);

tempLayers = [
    concatenationLayer(3,2,"Name","concat_6")
    reluLayer("Name","relu_7")
    convolution2dLayer([3 3],3,"Name","l_14","Padding","same")
    resize2dLayer("Name","resize-scale_7","GeometricTransformMode","half-pixel","Method","bilinear","NearestRoundingMode","round","Scale",[2 2])];
airfoilNet_1 = addLayers(airfoilNet_1,tempLayers);

% clean up helper variable
clear tempLayers;

airfoilNet_1 = connectLayers(airfoilNet_1,"dropout_1","leakyrelu_1");
airfoilNet_1 = connectLayers(airfoilNet_1,"dropout_1","concat_6/in2");
airfoilNet_1 = connectLayers(airfoilNet_1,"dropout_2","leakyrelu_2");
airfoilNet_1 = connectLayers(airfoilNet_1,"dropout_2","concat_5/in1");
airfoilNet_1 = connectLayers(airfoilNet_1,"dropout_3","leakyrelu_3");
airfoilNet_1 = connectLayers(airfoilNet_1,"dropout_3","concat_4/in2");
airfoilNet_1 = connectLayers(airfoilNet_1,"dropout_4","leakyrelu_4");
airfoilNet_1 = connectLayers(airfoilNet_1,"dropout_4","concat_3/in1");
airfoilNet_1 = connectLayers(airfoilNet_1,"dropout_5","leakyrelu_5");
airfoilNet_1 = connectLayers(airfoilNet_1,"dropout_5","concat_2/in1");
airfoilNet_1 = connectLayers(airfoilNet_1,"dropout_6","leakyrelu_6");
airfoilNet_1 = connectLayers(airfoilNet_1,"dropout_6","concat_1/in2");
airfoilNet_1 = connectLayers(airfoilNet_1,"batchnorm_6","concat_1/in1");
airfoilNet_1 = connectLayers(airfoilNet_1,"resize-scale_2","concat_2/in2");
airfoilNet_1 = connectLayers(airfoilNet_1,"resize-scale_3","concat_3/in2");
airfoilNet_1 = connectLayers(airfoilNet_1,"resize-scale_4","concat_4/in1");
airfoilNet_1 = connectLayers(airfoilNet_1,"resize-scale_5","concat_5/in2");
airfoilNet_1 = connectLayers(airfoilNet_1,"resize-scale_6","concat_6/in1");


% ------------------------------------------------------------------------
% CONVERT TO DLNET
dlnet = dlnetwork(airfoilNet_1);
end