function txt_to_tif(filepath)
    [path, filename, ext] = fileparts(filepath);
    raw_data = importdata(filepath);
    [Nr,Nc] = size(raw_data);
    Nx = Nr;
    Ny = Nr;
    Nz = Nc/Nr;
    hyper_noisy = reshape(raw_data, [Nx, Ny, Nz]);

    %output = strcat(filename, '.tif')
    %create_single_32_bit_tif(hyper_noisy, output)
    %im1 = imshow(hyper_noisy(:,:,1),'InitialMagnification',250);
    imwrite(hyper_noisy(:,:,1), [filename,'.tif']);
    for i=2:Nz
        %imshow(hyper_noisy(:,:,i), 'InitialMagnification',250);
        imwrite(hyper_noisy(:,:,i), [filename,'.tif'], 'WriteMode', 'append')
    end
end