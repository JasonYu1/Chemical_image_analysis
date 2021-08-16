function txt2tiff_32(file)
%file = 'C:\Users\User\Dropbox\My PC (LAPTOP-7BC0EJ3C)\Downloads\LS_LASSO\Step size0.0040_Dwell time10 Celegans_Jian_1040-100_800-20_2.45MHz_R_OBJ5_P23460_F23070_2_up1um.txt';
[filepath, name, ext] = fileparts(file);
img = single(dlmread(file));
frames = size(img,2)/size(img,1);
img_reshape = reshape(img,[size(img,1) size(img,1) frames]);
outputName = strcat(name,'.tif');
t = Tiff(outputName, 'w');

%raw_data = importdata(file);
%[Nr,Nc] = size(raw_data);
%Nx = Nr;
%Ny = Nr;
%Nz = Nc/Nr;
%img = im2uint8(reshape(raw_data, [Nx, Ny, Nz])); 
%frames = size(img,2)/size(img,1);
%img_avg = sum(reshape(img,[size(img,1) size(img,1) frames]),3)/frames;
%outputName = strcat(name,'.tif');
%t = Tiff(outputName, 'w');
    
    %
tagstruct.ImageLength = size(img, 1);
tagstruct.ImageWidth = size(img, 1);
tagstruct.Compression = Tiff.Compression.None;
tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
%tagstruct.SampleFormat = Tiff.SampleFormat.UInt;
tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
tagstruct.BitsPerSample = 32;
tagstruct.SamplesPerPixel = 1;
tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
%t.setTag(tagstruct);
%t.write(img_reshape);
for ii=1:size(img_reshape,3)
   setTag(t,tagstruct);
   write(t,img_reshape(:,:,ii));
   writeDirectory(t);
end
t.close();
end
