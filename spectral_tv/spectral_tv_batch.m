function out = spectral_tv(hyper_img, opts, out_stv_sigma)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% out = spectral_tv(hyper_img, opts)
% denoises image g by solving the following Spectral TV(STV) minimization
% problem:
% 
% min || f - g ||^2 + || w odots f ||_TV
% 
% where ||f||_TV = sqrt( a||Dxf||^2 + b||Dyf||^2 c||Dtf||^2),
% Dxf = f(x+1,y, t) - f(x,y,t),
% Dyf = f(x,y+1, t) - f(x,y,t),
% Dtf = f(x,y, t+1) - f(x,y,t) and
% odots = component-wise product (Hadamard product).
% 
% STV sets the weight parameter (w_{:,:,k} = sigma_k^1.1) using the 
% following noise level of each frame k estimated from selected smooth 
% region:
% 
% sigma_k = ( (1/|Omega|) sum_{i \in Omega} (f_i - mean(f))^2 )^(1/2)
% 
% 
% Input:  hyper_img      - the input hyperspectral image (can be 2d or 3d)
%         opts.tv_method - total variation method ('aniso' or 'iso') {'aniso'}
%         opts.rho_r     - initial penalty parameter for ||u-Df||   {2}
%         opts.rho_o     - initial penalty parameter for ||f-g-r|| {50}
%         opts.beta      - regularization parameter [a b c] for weighted TV norm {[1 1 0]}
%         opts.gamma     - update constant for rho_r {2}
%         opts.max_itr   - maximum iteration {20}
%         opts.alpha     - constant that determines constraint violation {0.7}
%         opts.tol       - tolerance level on relative change {1e-3}
%         opts.print     - print screen option {false}
%         opts.f         - initial  f {g}
%         opts.y1        - initial y1 {0}
%         opts.y2        - initial y2 {0}
%         opts.y3        - initial y3 {0}
%         opts.z         - initial  z {0}
%         ** default values of opts are given in { }.
% 
% Output: out.f          - output video
%         out.itr        - total number of iterations elapsed
%         out.relchg     - final relative change
%         out.Df1        - Dxf, f is the output video
%         out.Df2        - Dyf, f is the output video
%         out.Df3        - Dtf, f is the output video
%         out.y1         - Lagrange multiplier for Df1
%         out.y2         - Lagrange multiplier for Df2
%         out.y3         - Lagrange multiplier for Df3
%         out.rho_r      - final penalty parameter
%         out.sigma      - estimated noise level
%         out.w          - estimated weight parameter
% 
% Joon Hee Choi
% Copyright 2015
% Purdue University
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin<1
    error('not enough inputs, try again \n');
elseif nargin<2
    opts = [];
end

%sigma     = estimate_noise_level(hyper_img);
sigma     = out_stv_sigma;
w         = estimate_w(sigma, size(hyper_img));
opts.w    = w;

out       = deconvtvl2(hyper_img, 1, 1, opts);
out.sigma = sigma;
out.w     = w;

end
