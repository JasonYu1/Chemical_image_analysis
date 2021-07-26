function rmse = quality_check(filepath, denoise_file)
  A = imread(filepath);
  ref = imread(denoise_file);
  rmse = sqrt(mean((A(:)-ref(:)).^2));
end