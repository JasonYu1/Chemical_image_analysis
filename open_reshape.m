function y = open_reshape(file)

fid = fopen(file);
[Nx, Ny] = size(imread(file));
Nz = length(imfinfo(file));
%y = reshape(fread(fid,181*217*181),[181 217 181])/255;
y = reshape(fread(fid,Nx*Ny*Nz),[Nx Ny Nz])/255;
fclose(fid);
