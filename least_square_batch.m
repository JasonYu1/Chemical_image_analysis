function least_square_batch(sigma, input, original_filename, reference, normalize , peak_low, peak_high, raman_low, raman_high, sparsity_level)
%% Load data
filepath = input;

[path, filename, ext] = fileparts(filepath);

if ext == '.txt'
    raw_data = importdata(filepath);
    [Nr,Nc] = size(raw_data);
    Nx = Nr;
    Ny = Nr;
    Nz = Nc/Nr;
    y = reshape(raw_data, [Nx,Ny,Nz]);
elseif ext == '.tif'
    y = im2double(loadtiff(filepath));
    [Nx Ny Nz] = size(y);
end


%% Load pure chemicals
rpf = (raman_high-raman_low)/(peak_high-peak_low);
%rpf =(2994-2913)/(76-40); % Use DMSO to calibrate
Raman_shift = linspace(raman_low-peak_low*rpf, raman_high+(100-peak_high)*rpf,Nz);
%Raman_shift = linspace(2913-40*rpf, 2994+22*rpf,100)
%load 'Pure_chemicals.mat'

%load reference

%file = matfile('Pure_chemicals.mat');
file = matfile(reference);
varlist = who(file);
n = size(varlist);

dimension = size(file.(varlist{1}));
var = zeros(dimension(1), 1, n(1));
for i=1:n(1)
    value = file.(varlist{i});
    var(:,:,i) = ( value - min(value) ) ./ (max(value) - min(value));
end

% Normalize TAG and BSA
%BSA_n = ( BSA - min(BSA) ) ./ (max(BSA) - min(BSA));
%TAG_n = ( TAG - min(TAG) ) ./ (max(TAG) - min(TAG));

figure;
hold on
for i=1:n(1)
    plot(Raman_shift, var(:,:,i), 'Linewidth',1, 'DisplayName', char(varlist(i)));
end
hold off
legend show
mkdir ls_chemical_maps
title 'Spectral profiles';
set(gcf, 'Color','#F0F0F0');
set(gcf, 'InvertHardCopy', 'off');
saveas(gcf, 'ls_chemical_maps/pure_chemical_spectra.png');
%plot(Raman_shift, BSA_n, 'Linewidth',1);
%hold on
%plot(Raman_shift, TAG_n, 'Linewidth',1);
%hold off
%legend('Protein (BSA)','Lipid (TAG)')


%% Subtract background

y_sum = squeeze(mean(y,3));
figure; histogram(y_sum);
saveas(gcf, 'ls_chemical_maps/histogram.png');



%BGmask=zeros(size(y_sum));
%BG=find(y_sum < 0.00);
%BG = find(y_sum < mean(sigma)+ 3*std(sigma));
%size(y_sum)
%mean(sigma)
%mean(sigma)+ 3*std(sigma)
%BGmask(BG)=1;
%figure;imagesc(BGmask)
%saveas(gcf, 'ls_chemical_maps/bgmask.png');

%BG_spectrum = zeros(Nz,1);
%for i=1:1:Nz
%    y_temp = y(:,:,i);
%    BG_spectrum(i) = mean(y_temp(BGmask == 1)); 
%end
%BG_spectrum = BG_spectrum';
%figure;plot(BG_spectrum);
%saveas(gcf, 'ls_chemical_maps/bgspectrum.png');

%sigma = estimate_noise_level(y);
%figure;plot(sigma);

y_sub = zeros(size(y));
for i=1:1:Nz
    %y_sub(:,:,i) = y(:,:,i) - BG_spectrum(i);
    y_sub(:,:,i) = y(:,:,i) - sigma(i);
    
end

if normalize == 1
    y_sub(:,:,:) = (y_sub(:,:,:)-min(y_sub(:,:,:),[],'all'))/(max(y_sub(:,:,:),[],'all')-min(y_sub(:,:,:),[],'all'));
end

%% LS-LASSO unmixing
k = n(1);  % Set number of components to 2
for i=1:n(1)
    ref(:,i) = var(:,:,i);
end
%ref = var;
%ref = [BSA_n,TAG_n];    % Generate spectral reference matrix
C = zeros(Nx,Ny,k);     % Preallocate an empty data matrix to store unmixing map
L = sparsity_level;
% L = 5e-2;               % Set sparsity level (\lambda), if 0 then LS fitting

for i = 1:Nx
    for j = 1:Ny
        y_single_pixel = reshape(y_sub(i,j,:),[Nz,1]);
        %if isequal(filename, 'denoise_stv') == 1
        %    y_single_pixel = reshape(y(i,j,:),[Nz,1]);
        %elseif isequal(filename, 'denoise_bm4d') == 1
        %    y_single_pixel = reshape(y(i,j,:),[Nz,1]);  
        %else
        %    y_single_pixel = reshape(y_sub(i,j,:),[Nz,1]);
        %end
        %c_single_pixel = inv(ref'*ref)*ref'*y_single_pixel;
        c_single_pixel = lasso(ref,y_single_pixel,'lambda',L,...
            'MaxIter',1e5, 'Alpha', 1);    
        %c_single_pixel = ridge(y_single_pixel, ref, 5e-3);
        C(i,j,:) = reshape(c_single_pixel,[1,1,k]);
    end
end

%% Quick check unmixing quality

disp_min = prctile(reshape(C(:,:,1),[Nx*Ny,1]),0.4);
disp_max = prctile(reshape(C(:,:,1),[Nx*Ny,1]),99.7);

figure;
clims = [disp_min disp_max];
for i=1:n(1)
    subplot(1,n(1),i);imagesc(C(:,:,i),clims); colormap bone; axis off; axis square
end
saveas(gcf, 'ls_chemical_maps/unmixing_quality_check.png');
%subplot(1,2,1);imagesc(C(:,:,1),clims); colormap bone; axis off; axis square
%subplot(1,2,2);imagesc(C(:,:,2),clims); colormap bone; axis off; axis square

%% Output as txt file

output_ext   = '.txt';
opt_filepath = 'ls_chemical_maps/';
%Protein_map  = C(:,:,1);
%TAG_map      = C(:,:,2);

for i=1:n(1)
    out_filename = [char(varlist(i)), '_lambda_', num2str(L) '_',filename, output_ext];
    dlmwrite([opt_filepath, out_filename], C(:,:,i), 'delimiter','\t');
    figure;
    imshow(C(:,:,i));
    out_file_tif = [original_filename, '_', char(varlist(i)), '.tif'];
    set(gcf, 'Color','#F0F0F0');
    set(gcf, 'InvertHardCopy', 'off');
    saveas(gcf, [opt_filepath, out_file_tif]);
end
%protein_out_filename = ['Protein_lambda_', num2str(L) '_',filename, '_Protein', output_ext];
%TAG_out_filename = ['Lipid_lambda_', num2str(L) '_',filename, '_TAG', output_ext];

%dlmwrite([opt_filepath, protein_out_filename], Protein_map, 'delimiter','\t');
%dlmwrite([opt_filepath, TAG_out_filename], TAG_map, 'delimiter','\t');
end
