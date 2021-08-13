function rmse = quality_check(filepath, denoise_file, dtype)
  A = imread(filepath);
  ref = imread(denoise_file);
  if strcmp(dtype, 'float32') == 1
    ref = ref(:,:, 1);
  end
  rmse = sqrt(mean((A(:)-ref(:)).^2));
end