function show_stv(hyper_noisy, denoisedimage, num_frame)

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
im1 = imshow(denoisedimage(:,:,1),'InitialMagnification',250);
imwrite(denoisedimage(:,:,1), 'denoise_stv/denoise_stv.tif');
for i=2:num_frame
    %imshow(denoisedimage(:,:,i), 'InitialMagnification',250);
    imwrite(denoisedimage(:,:,i), 'denoise_stv/denoise_stv.tif', 'WriteMode', 'append');
end

end
