function make_beta_array(tv_method, rho_r, rho_o, a, b, c, gamma, max_itr, alpha, tol, hyper_noisy, K, input_type)

opts.tv_method = tv_method;
opts.rho_r = rho_r;
opts.rho_o = rho_o;
opts.beta = [a b c];
opts.gamma = gamma;
opts.max_itr = max_itr;
opts.alpha = alpha;
opts.tol = tol;
out_stv = spectral_tv(hyper_noisy, opts);
denoisedimage = out_stv.f;
show_stv(hyper_noisy, denoisedimage, K, input_type);

end