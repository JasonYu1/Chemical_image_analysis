function mcr(input, reference, peak_low, peak_high, raman_low, raman_high, sparsity_level, augmentation, itr)
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
mkdir mcr_chemical_maps
title 'Spectral profiles';
set(gcf, 'Color','#F0F0F0');
set(gcf, 'InvertHardCopy', 'off');
saveas(gcf, 'mcr_chemical_maps/pure_chemical_spectra.png');
%plot(Raman_shift, BSA_n, 'Linewidth',1);
%hold on
%plot(Raman_shift, TAG_n, 'Linewidth',1);
%hold off
%legend('Protein (BSA)','Lipid (TAG)')


%% Subtract background

y_sum = squeeze(mean(y,3));
figure; histogram(y_sum);
saveas(gcf, 'mcr_chemical_maps/histogram.png');

sigma = background_noise_level(y);

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

%% LS-LASSO unmixing
k = n(1);  % Set number of components to 2
for i=1:n(1)
    ref(:,i) = var(:,:,i);
end
%ref = var;
%ref = [BSA_n,TAG_n];    % Generate spectral reference matrix
L = sparsity_level;
% L = 5e-2;               % Set sparsity level (\lambda), if 0 then LS fitting
augnum  = augmentation*Nx*Ny;         % Number of datapoints for augmentation, default is 0.5*NxNy
iter    = itr;                 % Number of iterations for ADMM
tic
[C_2D, S]  = ALS_aug( y_sub, ref, augnum, iter);
toc

C = reshape(C_2D,[Ny,Nx,k]);


%% Quick check unmixing quality

disp_min = prctile(reshape(C(:,:,1),[Nx*Ny,1]),0.4);
disp_max = prctile(reshape(C(:,:,1),[Nx*Ny,1]),99.7);

figure;
clims = [disp_min disp_max];
for i=1:n(1)
    subplot(1,2,i);imagesc(C(:,:,i),clims); colormap bone; axis off; axis square
end
saveas(gcf, 'mcr_chemical_maps/unmixing_quality_check.png');
%subplot(1,2,1);imagesc(C(:,:,1),clims); colormap bone; axis off; axis square
%subplot(1,2,2);imagesc(C(:,:,2),clims); colormap bone; axis off; axis square

figure;
hold on
for i = 1:k
    plot(Raman_shift, S(:,i), 'Linewidth',1);
end
hold off
title 'Spectral profiles after MCR ALS'
set(gcf, 'Color','#F0F0F0');
set(gcf, 'InvertHardCopy', 'off');
saveas(gcf, 'mcr_chemical_maps/new_spectral_profiles.png');
%% Output as txt file

output_ext   = '.txt';
opt_filepath = 'mcr_chemical_maps/';
%Protein_map  = C(:,:,1);
%TAG_map      = C(:,:,2);

for i=1:n(1)
    out_filename = [char(varlist(i)), '_lambda_', num2str(L) '_',filename, output_ext];
    dlmwrite([opt_filepath, out_filename], C(:,:,i), 'delimiter','\t');
    figure;
    imshow(C(:,:,i));
    out_file_tif = [char(varlist(i)), '.tif'];
    set(gcf, 'Color','#F0F0F0');
    set(gcf, 'InvertHardCopy', 'off');
    saveas(gcf, [opt_filepath, out_file_tif]);
end
%protein_out_filename = ['Protein_lambda_', num2str(L) '_',filename, '_Protein', output_ext];
%TAG_out_filename = ['Lipid_lambda_', num2str(L) '_',filename, '_TAG', output_ext];

%dlmwrite([opt_filepath, protein_out_filename], Protein_map, 'delimiter','\t');
%dlmwrite([opt_filepath, TAG_out_filename], TAG_map, 'delimiter','\t');
end
