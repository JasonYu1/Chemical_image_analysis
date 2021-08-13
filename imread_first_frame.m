function y = imread_first_frame(filepath)
    x = imread(filepath);
    y = x(:,:,1);
end