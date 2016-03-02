function [ calibration_points ] = FindCalibrationPointsGBVS(input_image, threshold_level)
%Runs gbvs on the input image to get a saliency map.  Using the saliency
%map, this function thresholds it, finds the centroids, and returns the
%points in the scene upon which the user must focus.
%
n = 1;

if nargin < 2
   threshold_level = 0.5; 
end

result = gbvs(input_image);
sal_map = result.master_map_resized;

paramRGB = default_signature_param;
sal_map = signatureSal(input_image, paramRGB);

roi = sal_map > threshold_level;

stats = regionprops(roi, 'Centroid');

% figure;
% imshow([sal_map, roi]);
% hold on;

calibration_points = [];
for i = 1 : size(stats)
    h = scatter(stats(i).Centroid(1), stats(i).Centroid(2), 'rx');
    hc = get(h, 'Children');
    set(hc, 'MarkerSize', 30);
    calibration_points = [calibration_points; ...
        [stats(i).Centroid(1), stats(i).Centroid(2), sal_map(floor(stats(i).Centroid(2)), floor(stats(i).Centroid(1)))]];
end

% average_point = [0, 0];
% for i = 1 : size(result, 1)
%    average_point = average_point + [(result(i, 1) + result(i, 3) / 2), (result(i, 2) + result(i, 4) / 2)];
% end
% average_point = average_point / size(result, 1);
% scatter(average_point(1), average_point(2));
end

