function out = mse(x,y)
out = mean( (x(:)-y(:)).^2 );