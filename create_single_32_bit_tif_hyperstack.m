function create_single_32_bit_tif_hyperstack(denoisedimage, filename)
A = im2single(denoisedimage);
t = Tiff(filename, 'w');
tagstruct.ImageLength = size(A, 1);
tagstruct.ImageWidth = size(A, 2);
tagstruct.Compression = Tiff.Compression.None;
tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
tagstruct.BitsPerSample = 32;
%tagstruct.SamplesPerPixel = size(A,3);
tagstruct.SamplesPerPixel = size(A, 3);
tagstruct.ExtraSamples = Tiff.ExtraSamples.Unspecified;
%tagstruct.SubFileType = Tiff.SubFileType.Page;
tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
setTag(t,tagstruct);
write(t,A);
t.close();
end