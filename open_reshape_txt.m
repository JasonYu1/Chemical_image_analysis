function y = open_reshape_txt(filepath)
    raw_data = importdata(filepath);
    [Nr,Nc] = size(raw_data);
    Nx = Nr;
    Ny = Nr;
    Nz = Nc/Nr;

    y = reshape(raw_data, [Nx, Ny, Nz]);
end
