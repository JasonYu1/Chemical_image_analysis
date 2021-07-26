clear all
close all
clc
%%
addpath(genpath('./spectral_tv/'));

% Set the number of rows, columns and frames.
M           = 200;
N           = 200;
K           = 128;

file_name2   = '../data/true_Prostate_tumor_Rchan_891nm_30mW_1040nm_200mW_nframe_10_6rods_7_9.tif';
hyper_true  = read_hyperdata(file_name2, M, N, K);
[rows, cols, frames] = size(hyper_true);

file_name   = '../data/noisy_Prostate_tumor_Rchan_891nm_30mW_1040nm_200mW_nframe_10_6rods_7_9.tif';
hyper_noisy  = read_hyperdata(file_name, M, N, K);
[rows, cols, frames] = size(hyper_noisy);

% Spectral Total Variation
opts.beta   = [0.65 0.65 0.4];
runtime     = tic;
out_stv     = spectral_tv(hyper_noisy, opts);
runtime_stv = toc(runtime);
sigma_est   = out_stv.sigma;
%psnr_stv    = psnr(hyper_true, out_stv.f);

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

pause(0.4);

end

subplot(1,3,1);
imshow(hyper_true(:,:,40),'InitialMagnification',250);
title('True image');  

subplot(1,3,2);
imshow(hyper_noisy(:,:,40),'InitialMagnification',250);
title('Noisy image');  

subplot(1,3,3);
imshow(denoisedimage(:,:,40));
title('Denoised image');  

%% Output 3D image 
t1 = ['STV_noisy_Prostate_tumor_Rchan_891nm_30mW_1040nm_200mW_nframe_10_6rods_7_9.tif'];
eval(sprintf(['delete ' t1]));

y_n_denoised_16bit = uint16(65535.*denoisedimage);

for ii=1:K
    imwrite((denoisedimage(:, :, ii)), t1, 'WriteMode', 'append','Compression','None');
    
end