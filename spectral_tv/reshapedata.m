function yData = reshapedata(y,win,part,method)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% yData = reshapedata(y, win, part, method)
% converts all overlapping patches of the input image into a data
% volume.
%
% y      - input image
% win    - patch radius (diameter is 2*win+1)
% part   - a string, either 'intensity', or 'spatial'
% method - a string, options are 'symmetric', 'replicate', 'circular'
%
% Stanley Chan
% Purdue 2014-10-15
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[M, N] = size(y);

% Padding Method
switch method
    case 'symmetric'
        tmppad = padarray(y,[win win],'symmetric');
    case 'replicate'
        tmppad = padarray(y,[win win],'replicate');
    case 'circular'
        tmppad = padarray(y,[win win],'circular');
    case 'zero'
        tmppad = padarray(y,[win win]);
    case 'none'
        tmppad = y;
        M = M - 2*win;
        N = N - 2*win;
    otherwise
        error('please input correct method \n');
end
        
% Rearrange y into a data volume
Y = zeros(M,N,(2*win+1)^2);
for i=1:2*win+1
    for j=1:2*win+1
        k = sub2ind([2*win+1 2*win+1], i, j);
        tmp = tmppad(i:i+M-1, j:j+N-1);
        Y(:,:,k) = tmp;
    end
end

% Pad spatial coordinates if needed
switch part
    case 'intensity'
        yData = reshape(shiftdim(Y,2),(2*win+1)^2, M*N);
    case 'spatial'
        [Xgrid, Ygrid] = meshgrid(1:N,1:M);
        yData = cat(3,Ygrid,Xgrid,Y);
        yData = reshape(shiftdim(yData,2),(2*win+1)^2+2, M*N);
    otherwise
        error('please specify either "intensity" or "spatial" \n');
end

end
