function create_single_32_bit_tif(denoisedimage, filename)
A = im2single(denoisedimage);
t = Tiff(filename, 'w');
tagstruct.ImageLength = size(A, 1);
tagstruct.ImageWidth = size(A, 2);
tagstruct.Compression = Tiff.Compression.None;
tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
%tagstruct.Photometric = Tiff.Photometric.RGB;
tagstruct.ExtraSamples = Tiff.ExtraSamples.Unspecified;
tagstruct.BitsPerSample = 32;
tagstruct.SamplesPerPixel = size(A,3);
tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
t.setTag(tagstruct);
t.write(A);
t.close();
end