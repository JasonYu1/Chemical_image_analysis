clear all
close all
clc
%%
addpath(genpath('./spectral_tv/'));

sel_exp     = 2;  % select an experiment (1 or 2)

% Set the number of rows, columns and frames.
M           = 200;
N           = 200;
K           = 40;

% Load a hyperspectral image (DMSO100%)

file_name2   = '../data/beads_true.tif';
hyper_true  = read_hyperdata(file_name2, M, N, K);
[rows, cols, frames] = size(hyper_true);

file_name   = '../data/beads_noisy.tif';
hyper_noisy  = read_hyperdata(file_name, M, N, K);
[rows, cols, frames] = size(hyper_noisy);

% Add noise in the hyperspectral image
%if sel_exp==1
 %   tmp        = read_hyperdata('../data/DMSO10%.tif', M, N, K);
  %  sigma_true = estimate_noise_level(tmp);
%elseif sel_exp==2
 %   sigma_true = 0.005:0.005:(0.005*frames);
%end

%hyper_noisy = zeros(rows, cols, frames);
%for i=1:frames
 %   hyper_noisy(:,:,i) = hyper_true(:,:,i) + sigma_true(i)*randn(rows, cols);
%end


% Spectral Total Variation
opts.beta   = [0.65 0.65 0.4];
runtime     = tic;
out_stv     = spectral_tv(hyper_noisy, opts);
runtime_stv = toc(runtime);
sigma_est   = out_stv.sigma;
psnr_stv    = psnr(hyper_true, out_stv.f);

denoisedimage=out_stv.f;


for k=1:40
subplot(1,3,1);
imshow(hyper_true(:,:,k),'InitialMagnification',250);
title('True image');

subplot(1,3,2);
imshow(hyper_noisy(:,:,k),'InitialMagnification',250);
title('Noisy image');

subplot(1,3,3);
imshow(denoisedimage(:,:,k),'InitialMagnification',250);
title('Denoised image');

pause(0.2);

end

subplot(1,3,1);
imshow(hyper_true(:,:,20),'InitialMagnification',250);
title('True image');  

subplot(1,3,2);
imshow(hyper_noisy(:,:,20),'InitialMagnification',250);
title('Noisy image');  

subplot(1,3,3);
imshow(denoisedimage(:,:,20));
title('Denoised image');  




% Original Total Variation
%mu          = 1;
%opts.w      = mean(out_stv.w(:));
%opts.beta   = [1 1 0.1];
%runtime     = tic;
%out_tv      = deconvtvl2(hyper_noisy, 1, mu, opts);
%runtime_tv  = toc(runtime);
%psnr_tv     = psnr(hyper_true, out_tv.f);




% Print PSNRs between true and denoised images.
%fprintf('Method: spectral tv, \t psnr: %6.4f, \t runtime: %6.4f\n', psnr_stv, runtime_stv);
%fprintf('Method: original tv, \t psnr: %6.4f, \t runtime: %6.4f\n', psnr_tv, runtime_tv);

% Plot the true and estimeated noise level.
%if sel_exp==1 || sel_exp==2
 %   figure;
  %  plot(sigma_true, 'LineWidth', 2, 'Color', 'g');
   % hold on;
    %plot(sigma_est, 'LineWidth', 2, 'Color', 'r');
    %hold off;
    %xlabel('frame');
    %ylabel('noise standard deviation');
    %legend('True', 'Estimated', 'Location', 'best');
%end
