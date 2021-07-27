function y_est_to_tif(y_est, K, filetype)
    % if filetype == '.txt'
    imwrite(y_est(:,:,1), ['denoise_bm4d/', 'denoise_bm4d.tif']);
    for i=2:K
        imwrite(y_est(:,:,i), ['denoise_bm4d/', 'denoise_bm4d.tif'], 'WriteMode', 'append');
    end
    %elseif filetype == '.tif'
    %    [A, B, C] = size(y_est);
    %    new = zeros(A, B, C);
    %    for i=1:A
    %        for j=1:B
    %            for k=1:C
    %                new(j,i,k) = y_est(i,j,k);
    %                % new(i,j,k) = y_est(i,j,k);
    %            end
    %        end
    %    end
    %    imwrite(new(:,:,1), ['denoise_bm4d/', 'denoise_bm4d.tif']);
    %    for i=2:K
    %        imwrite(new(:,:,i), ['denoise_bm4d/', 'denoise_bm4d.tif'], 'WriteMode', 'append');
    %    end
        
    %end
end