function w = estimate_w(sigma, ySize)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% w = estimate_w(sigma, ySize)
% sets the weight parameter (w_{:,:,k} = sigma_k^1.1) from the given sigma.
% 
% Input:  sigma  - the noise standard deviation of selected region
%         ySigze - the size of an image
% Output: w      - the weight parameter
% 
% Stanley Chan
% Copyright 2015
% Purdue University
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

M = ySize(1);
N = ySize(2);
if numel(ySize)==3
    K = ySize(3);
else
    K = 1;
end

w = reshape(sigma.^1.1, [1 1 K]);
w = repmat(w, [M N 1]);
