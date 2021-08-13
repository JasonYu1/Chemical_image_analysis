function sigma = estimate_noise_level(y)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sigma = estimate_noise_level(y)
% estimates the following noise level of each frame k 
% from selected smooth region (Omega):
% 
% sigma = ( (1/|Omega|) sum_{i \in Omega} (f_i - mean(f))^2 )^(1/2)
% 
% Input:  y      - the input image, can be gray scale, color, or video
% Output: sigma  - the noise standard deviation of selected region
% 
% Stanley Chan
% Copyright 2015
% Purdue University
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K        = size(y,3);
%size(y)
% Select a region
if length(size(y)) == 2
    imshow(y(:,:,1), [],'InitialMagnification',250);
elseif length(size(y)) == 3
    imshow(y(:,:,round(K/2)), [],'InitialMagnification',250);
end

h        = imrect;
position = wait(h);
close all;

% Obtain the coordinates of the selected region
xmin     = round(position(1));
ymin     = round(position(2));
xwidth   = round(position(3));
ywidth   = round(position(4));
xmax     = xmin+xwidth;
ymax     = ymin+ywidth;

% Calculate the noise standard deviation of the region
K        = size(y,3);
win      = 5;
sigma    = zeros(1,K);
for k=1:K
    Y    = y(ymin:ymax, xmin:xmax, k);
    Ycol = reshapedata(Y, win, 'intensity', 'symmetric');
    sigma(k) = sqrt(mean(var(Ycol)));
end
close all
