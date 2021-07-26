function sixteen_to_eight(File_Name)
%Multiple image tiff conversion%

%File_Name = "C:/Users/User/Dropbox/My PC (LAPTOP-7BC0EJ3C)/Documents/ProstateAndLungTissue_Polygon SRS_Fingerprint_Hyperspectrum/Prostate_area4\HighSNR_ProstateTumor_area4_5.tif";
Image_Data = imfinfo(File_Name);
Number_Of_Images = length(Image_Data);


Tiff_Structure = struct('Image_File',[]);  

for Image_Index = 1: Number_Of_Images
    
      Image = imread(File_Name,Image_Index);
      Uint8_Image = im2uint8(Image);

      %For more information and plotting individual images%
      Tiff_Structure(Image_Index).Image_File = Uint8_Image;
      
      %Saving the converted images to one tiff file%
      imwrite(Uint8_Image,'Converted_Image.tif','WriteMode','append');

end
end