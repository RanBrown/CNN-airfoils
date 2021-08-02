function array_max = absMax(array)
% Calculates absolute maximum value of first two dimensions of array,
% assuming that channel and sample are fixed.

absVal = abs(array);

vecMax = max(absVal, [], 1);
array_max = max(vecMax);

end
