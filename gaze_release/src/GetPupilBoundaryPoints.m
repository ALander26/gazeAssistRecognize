function [x,y] = GetPupilBoundaryPoints(image, pupil_thresh, reflection_thresh)
    
   
    denoised_image = AnisotropicDiffusion(image(:,:,1), 7);
    bw_denoised_image = im2bw(denoised_image, pupil_thresh);
    reflection_im = im2bw(image(:,:,1), reflection_thresh);
    bw_denoised_image = bw_denoised_image - reflection_im;
    %imshow([bw_denoised_image, reflection_im])
    
    LB = 300; %TODO(perra): was 10.
    UB = 16000; %changed by rohit was 10000 
    
    inv_bw = ~bw_denoised_image;
    Iout = xor(bwareaopen(inv_bw, LB), bwareaopen(inv_bw, UB));
    %imshow([inv_bw, Iout]);
    %figure;
    bw_denoised_image = ~Iout;
    F = [0 1 0; 1 -4 1; 0 1 0];
    outline = conv2(double(bw_denoised_image), F);
    outline = outline(4:end-4, 4:end-4);
    %output = del2(double(bw_denoised_image));
    %commented after parfor
   % imshow(output);

    [ya,xa] = ind2sub(size(outline), find(outline > 0));
    K = convhull(xa,ya);

    x = [];
    y = [];
    for i = 1 : size(K)
        x = [x; xa(K(i))]; 
        y = [y; ya(K(i))]; 
    end
end

