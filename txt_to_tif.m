function txt_to_tif(filepath)
    raw_data = importdata(filepath);
    [Nr,Nc] = size(raw_data);
    Nx = Nr;
    Ny = Nr;
    Nz = Nc/Nr;
    hyper_noisy = reshape(raw_data, [Nx, Ny, Nz]);
    
    %im1 = imshow(hyper_noisy(:,:,1),'InitialMagnification',250);
    imwrite(hyper_noisy(:,:,1), 'raw.tif');
    for i=2:Nz
        %imshow(hyper_noisy(:,:,i), 'InitialMagnification',250);
        imwrite(hyper_noisy(:,:,i), 'raw.tif', 'WriteMode', 'append')
    end
end