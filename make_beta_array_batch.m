function [PSNR, SSIM] = make_beta_array(out_stv_sigma, tv_method, rho_r, rho_o, a, b, c, gamma, max_itr, alpha, tol, hyper_noisy, K, input_type, filename)

opts.tv_method = tv_method;
opts.rho_r = rho_r;
opts.rho_o = rho_o;
opts.beta = [a b c];
opts.gamma = gamma;
opts.max_itr = max_itr;
opts.alpha = alpha;
opts.tol = tol;
out_stv = spectral_tv_batch(hyper_noisy, opts, out_stv_sigma);
denoisedimage = out_stv.f;
PSNR = psnr(hyper_noisy, denoisedimage);
SSIM = ssim(hyper_noisy, denoisedimage);
show_stv(hyper_noisy, denoisedimage, K, input_type, filename);

end