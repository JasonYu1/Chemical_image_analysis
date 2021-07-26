clear all
phantom           = 't1_icbm_normal_1mm_pn0_rf0.rawb';
crop_phantom = 1;
fid = fopen(phantom);
y = reshape(fread(fid,181*217*181),[181 217 181])/255;
fclose(fid);
if crop_phantom
    y = y(51:125,51:125,51:125);
end
estimate_sigma = 0;
variable_noise = 0;
estimate_sigma = estimate_sigma>0 || variable_noise>0;
