function [ calibration_points ] = FindCalibrationPointsSim(input_image)
% get the centroid of the object in the simulated image (in simple simulated images)
%
if(size(input_image,3)>1)
input_image = rgb2gray(input_image);
end

level = graythresh(input_image);
bw = im2bw(input_image,level);
[b, l] = bwboundaries(~bw, 'noholes');
stats = regionprops(l, 'Area', 'Centroid');


calibration_points = [stats.Centroid(1) stats.Centroid(2)];
% figure;
% imshow(input_image);
% hold on;
% for i = 1 : size(calibration_points, 1)
%     scatter(calibration_points(i, 1), calibration_points(i, 2), 'bo');
% end


end
