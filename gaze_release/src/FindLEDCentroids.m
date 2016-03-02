function [ centroids ] = FindLEDCentroids(im, reflection_thresh, OFFSET_X, OFFSET_Y)

if nargin < 3
   OFFSET_X = 0;
   OFFSET_Y = 0;
elseif nargin < 4
   OFFSET_Y = 0;
end

bw_im = im2bw(im, reflection_thresh);
LB = 0;
UB = 250 ;%90; %TODO(perra): was 20, then 35
Iout = xor(bwareaopen(bw_im, LB), bwareaopen(bw_im, UB));
stats = regionprops(Iout, 'Centroid');
centroids = zeros(2, 3);

for i = 1 : 2
   s = stats(i);
   centroids(i, :) = [s.Centroid(1) + OFFSET_X, s.Centroid(2) + OFFSET_Y, 1];
end

% Make sure that the reflection with the largest y value comes second in the
% list.
if centroids(1, 2) > centroids(2, 2)
   swap = centroids(2, :);
   centroids(2, :) = centroids(1, :);
   centroids(1, :) = swap;
end
% imshow(im); hold on;
% for i = 1 : 2
%    scatter(centroids(i,1) - OFFSET_X, centroids(i,2) - OFFSET_Y, 'bo');
%    hold on;
% end

end

