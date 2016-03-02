% System params is a vector set up contain:
% [tx, ty, tz, rx, ry, rz, d_eye_to_front_camera, corneal_radius, n1, alpha_eye, beta_eye]
function [ C_out, P_out, sc, sp, sp_inf, gaze_slope, gaze_offset] = ProcessEyeFunc(centroids, image_pupil_center, system_params, num, last_C, last_P, K, scene_K, led_positions, pupil_thresh, OFFSET_X, OFFSET_Y)

scenecam_tx = system_params(1);
scenecam_ty = system_params(2);
scenecam_tz = system_params(3);
scenecam_rx = system_params(4);
scenecam_ry = system_params(5);
scenecam_rz = system_params(6);
distance_to_eye = system_params(7);
corneal_radius = system_params(8);
% n1 = system_params(9);
% alpha_eye = system_params(10);
% beta_eye = system_params(11);
alpha_eye = system_params(9);
beta_eye = system_params(10);
%distance_from_C_to_P = system_params(11);

distance_from_C_to_P = 4.5;
n1 = 1.3375;

Rx = [1                    0                    0;
    0                    cosd(scenecam_rx)    -sind(scenecam_rx);
    0                    sind(scenecam_rx)    cosd(scenecam_rx); ];

Ry = [cosd(scenecam_ry)    0                    sind(scenecam_ry);
    0                    1                    0;
    -sind(scenecam_ry)   0                    cosd(scenecam_ry); ];

Rz = [cosd(scenecam_rz)    -sind(scenecam_rz)   0;
    sind(scenecam_rz)    cosd(scenecam_rz)    0;
    0                    0                    1];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Move image coordinates to world coordinates and normalize
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
reflection_world = K \ centroids';
reflection_world(:,1) = reflection_world(:, 1) / norm(reflection_world(:,1));
reflection_world(:,2) = reflection_world(:, 2) / norm(reflection_world(:,2));
reflection_world = reflection_world * distance_to_eye;

% reflection_world = centroids/K;
% reflection_world(:,1) = reflection_world(:, 1) / norm(reflection_world(:,1));
% reflection_world(:,2) = reflection_world(:, 2) / norm(reflection_world(:,2));
% reflection_world = reflection_world' * distance_to_eye;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Find the center of the corneal sphere, C
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C = FindCornealCenter(led_positions, reflection_world, corneal_radius, distance_to_eye);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Find the point of refraction, rj
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rj = FindPointOfRefraction(K, image_pupil_center, distance_to_eye, OFFSET_X, OFFSET_Y);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Find the pupil center, P
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
P = FindPupilCenter(C, rj, distance_from_C_to_P, n1, alpha_eye, beta_eye);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Display the scene image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%figure;
%imshow([full_user_image, scene_image]);
%hold on;

ic = K * C;
ic = ic / ic(3);
ip = K * P;
ip = ip / ip(3);

% hold on;
% scatter(ic(1), ic(2), 'g');
% scatter(ip(1), ip(2), 'y');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Change to the frame of reference of the scene camera
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if num > 2
    C = (C * 0.5 + last_C * 0.5);
    P = (P * 0.5 + last_P * 0.5);
end

C_out = C;
P_out = P;

P = P - [scenecam_tx;scenecam_ty;scenecam_tz];
C = C - [scenecam_tx;scenecam_ty;scenecam_tz];

Pf = Rx * P;
Pf = Ry * Pf;
Pf = Rz * Pf;

Cf = Rx * C;
Cf = Ry * Cf;
Cf = Rz * Cf;


c_to_p_unit_vec = Pf - Cf;
P_inf = Cf + c_to_p_unit_vec * 10000;

sc = scene_K * Cf;
sc = sc / sc(3);
sp = scene_K * Pf;
sp = sp / sp(3);
sp_inf = scene_K * P_inf;
sp_inf = sp_inf / sp_inf(3);

% [gaze_slope, gaze_offset] = LongLine(sp, sc, false, false);
gaze_slope = (sc(2) - sp(2)) / (sc(1) - sp(1));
gaze_offset = sp(2) - gaze_slope* sp(1);


end

