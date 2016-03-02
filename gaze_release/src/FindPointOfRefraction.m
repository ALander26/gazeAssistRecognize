function [ rj ] = FindPointOfRefraction(K, image_pupil_center, distance_to_eye, OFFSET_X, OFFSET_Y)
if nargin < 4 
   OFFSET_X = 0;
   OFFSET_Y = 0;
elseif nargin < 5
   OFFSET_Y = 0; 
end
pupil_center_image_world = K \ [image_pupil_center(1) + OFFSET_X; image_pupil_center(2) + OFFSET_Y; 1];
pupil_center_image_world = pupil_center_image_world / norm(pupil_center_image_world);
pupil_center_image_world = pupil_center_image_world * distance_to_eye;

rj = pupil_center_image_world;

end

