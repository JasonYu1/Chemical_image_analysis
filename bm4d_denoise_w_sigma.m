function bm4d_denoise_w_sigma(y, K, sigma, estimate_sigma, distribution, profile, do_wiener, verbose, variable_noise, noise_factor, input_type)

% generate noisy phantom
randn('seed',0);
rand('seed',0);
sigma = sigma/100;
if variable_noise==1
    disp(['Spatially-varying ',distribution,' noise in the range [',...
        num2str(sigma),',',num2str(noise_factor*sigma),']'])
    map = helper.getNoiseMap(y,noise_factor);
else
    disp(['Uniform ',distribution,' noise ',num2str(sigma)])
    map = ones(size(y));
end
eta = sigma*map;
if strcmpi(distribution,'Rice')
    z = sqrt( (y+eta.*randn(size(y))).^2 + (eta.*randn(size(y))).^2 );
else
    z = y + eta.*randn(size(y));
end

% perform filtering
disp('Denoising started')
[y_est, sigma_est] = bm4d(z, distribution, (~estimate_sigma)*sigma, profile, do_wiener, verbose);


% objective result
%ind = y>0;
%PSNR = 10*log10(1/mean((y(ind)-y_est(ind)).^2));
%SSIM = ssim_index3d(y*255,y_est*255,[1 1 1],ind);
%fprintf('Denoising completed: PSNR %.2fdB / SSIM %.2f \n', PSNR, SSIM)

% plot historgram of the estimated standard deviation
if K > 1
    if estimate_sigma
        helper.visualizeEstMap( y, sigma_est, eta );
    end
end

% show cross-sections
if K > 1
    helper.visualizeXsect( y, z, y_est );
end

if strcmp(input_type, 'uint16') == 1
    y_est = im2uint16(y_est);
    if K == 1
        [A, B, C] = size(y_est);
        new = zeros(B, A, C);
        for i=1:A
            for j=1:B
                for k=1:C
                    new(k,i,j) = y_est(i,j,k);
                    % new(i,j,k) = y_est(i,j,k);
                end
            end
        end
    end

    if K == 1
        imwrite(new(:,:,1), ['denoise_bm4d/', 'denoise_bm4d.tif']);
    end

    if K > 1
        imwrite(y_est(:,:,1), ['denoise_bm4d/', 'denoise_bm4d.tif']);
        for i=2:K
            imwrite(y_est(:,:,i), ['denoise_bm4d/', 'denoise_bm4d.tif'], 'WriteMode', 'append');
        end
    end
elseif strcmp(input_type, 'float32') == 1
    create_single_32_bit_tif(y_est, 'denoise_bm4d/denoise_bm4d.tif');


    if K == 1
        [A, B, C] = size(y_est);
        new = zeros(B, A, C);
        for i=1:A
            for j=1:B
                for k=1:C
                    new(k,i,j) = y_est(i,j,k);
                    % new(i,j,k) = y_est(i,j,k);
                end
            end
        end
    end

elseif strcmp(input_type, 'txt') == 1
    create_single_32_bit_tif(y_est, 'denoise_bm4d/denoise_bm4d.tif');
    create_single_32_bit_tif_hyperstack(y_est, 'denoise_bm4d/denoise_bm4d_32bit.tif');
else
    imwrite(y_est(:,:,1), ['denoise_bm4d/', 'denoise_bm4d.tif']);
    if K > 1
        for i=2:K
            imwrite(y_est(:,:,i), ['denoise_bm4d/', 'denoise_bm4d.tif'], 'WriteMode', 'append');
        end
    end
end



end



