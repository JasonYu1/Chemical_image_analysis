function show_stv(hyper_noisy, denoisedimage, num_frame, input_type)
K = num_frame;
%subplot(1,3,2);
figure
imshow(hyper_noisy(:,:,round(num_frame/2)),'InitialMagnification',250);
%title('Noisy image');
set(gcf, 'Color','#F0F0F0');
set(gcf, 'InvertHardCopy', 'off');
saveas(gcf, 'denoise_stv/stv_noise_sample.png');



figure
%subplot(1,3,3);
imshow(denoisedimage(:,:,round(num_frame/2)),'InitialMagnification',250);
%title('Denoised image');
set(gcf, 'Color','#F0F0F0');
set(gcf, 'InvertHardCopy', 'off');
saveas(gcf, 'denoise_stv/stv_denoise_sample.png');

%input_type

if strcmp(input_type, 'float32') == 1
    create_single_32_bit_tif(denoisedimage, 'denoise_stv/denoise_stv.tif');

elseif strcmp(input_type, 'txt') == 1
    create_single_32_bit_tif(denoisedimage, 'denoise_stv/denoise_stv.tif');
    create_single_32_bit_tif_hyperstack(denoisedimage, 'denoise_stv/denoise_stv_32bit.tif');

elseif strcmp(input_type, 'uint16') == 1
    denoisedimage = im2uint16(denoisedimage);
    imwrite(denoisedimage(:,:,1), 'denoise_stv/denoise_stv.tif');

    if K > 1
        for i=2:num_frame
            %imshow(denoisedimage(:,:,i), 'InitialMagnification',250);
            imwrite(denoisedimage(:,:,i), 'denoise_stv/denoise_stv.tif', 'WriteMode', 'append');
        end
    end


elseif strcmp(input_type, 'xxx') == 1
    imwrite(denoisedimage(:,:,1), 'denoise_stv/denoise_stv.tif');

    if K > 1
        for i=2:num_frame
            %imshow(denoisedimage(:,:,i), 'InitialMagnification',250);
            imwrite(denoisedimage(:,:,i), 'denoise_stv/denoise_stv.tif', 'WriteMode', 'append');
        end
    end
end





end
