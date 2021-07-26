function y = read_hyperdata(fname, M, N, K)

data_info = imfinfo(fname);
if nargin<2
    M = data_info(1).Height;
end
if nargin<3
    N = data_info(1).Width;
end
if nargin<4
    K = length(data_info);
end

x = zeros(M,N,K);
for k=1:K
    x(:,:,k) = imresize(imread(fname,k),[M N]);
end
minX = min(x(:)) - 1e-8;
maxY = max(abs(x(:) - minX));
y = x - minX;
y = y/maxY;
