function [reflections] = DetectReflections(image)
    denoised_image = AnisotropicDiffusion(image(:,:,1), 7);
    thresholded = im2bw(denoised_image, 0.6);
    %imshow(thresholded)
    
    ycbcr = rgb2ycbcr(image);
    y = ycbcr(:,:,1);
    cb = ycbcr(:,:,2);
    cr = ycbcr(:,:,3);
    %imshow([im2bw(image) im2bw(y) im2bw(cb) im2bw(cr)]);
    
    reflections = [];
end

