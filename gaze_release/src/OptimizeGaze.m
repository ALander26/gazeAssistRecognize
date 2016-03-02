% user_eye = imread('autocalib_ir_user1.png');
% full_user_image = imread('autocalib_ir_user1_full.png');
% scene_image = imread('autocalib_ir_scene1.png');
function result = OptimizeGaze(K, scene_K, led_positions, calib_points, centroids, pupil_centers, system_params, predict, DATASET,start_num, winsize, Niter,  DO_NOT_USE,scene_face_info,OFFSET_X,OFFSET_Y, user, scene, pupil_thresh, old_C, old_P,last_d_sc_sp_thresh, cols, d, c_dif, p_dif, user_eyes, scene_images)

fun = @(x) ProcessEyeFuncLinCost(K, scene_K, led_positions, calib_points, centroids, pupil_centers, x, predict,DATASET,start_num, winsize, DO_NOT_USE,scene_face_info,OFFSET_X,OFFSET_Y, user, scene,pupil_thresh, old_C, old_P,last_d_sc_sp_thresh, cols, d, c_dif, p_dif, user_eyes,scene_images );
% 
% optnew=optimset('Display','iter','PlotFcns',@optimplotfval, 'MaxIter', Niter, 'UseParallel', 'Always');
options = optimoptions('lsqnonlin', 'MaxIter', Niter, 'MaxFunEvals', 1000, 'TolFun', 1e-8, 'Display', 'Off');
result = lsqnonlin(fun, system_params,[],[],options);
% % result = fminunc(fun, system_params, optnew);


%c2-c1




