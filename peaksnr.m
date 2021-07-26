function y = peaksnr(A, ref)
    A = im2double(A);
    ref = im2double(ref); 
    y = psnr(A,ref);
end