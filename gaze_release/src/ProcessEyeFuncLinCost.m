% vec is [tx, ty, tz, rx, ry, rz, d_eye_to_front_camera, corneal_radius, n1, alpha_eye, beta_eye]
function [ d ] = ProcessEyeFuncLinCost(K, scene_K, led_positions, calib_points, centroids, pupil_centers, system_params, predict, DATASET,start_num, winsize,  DO_NOT_USE,scene_face_info,OFFSET_X,OFFSET_Y, user, scene, pupil_thresh, old_C, old_P,last_d_sc_sp_thresh, cols, d, c_dif, p_dif, user_eyes, scene_images )

count = 1;
for num = start_num : start_num + winsize + predict
    %for num = start_num : size(scene_face_info, 1) - 1
    if sum(DO_NOT_USE == num) > 0
        continue;
    end
    calib_point = calib_points{num+1-start_num};
    centroid = centroids{num+1-start_num};
    pupil_center = pupil_centers{num+1-start_num};
   
    %last_d_sc_sp_thresh = mean(d_sc_sp) - std(d_sc_sp);
    %         if last_d_sc_sp_thresh < 0.0001
    %             last_d_sc_sp = 13.8089
    %         end
    
    %Keep track of the pupil and corneal movement between frames.
    oC = old_C;
    oP = old_P;
    [old_C, old_P, sc, sp, slope, offset] = ProcessEyeFunc(centroid, pupil_center, system_params, num, old_C, old_P, K, scene_K, led_positions, pupil_thresh,OFFSET_X, OFFSET_Y );
    %     dist_perp = real(abs(calib_point(2) - slope * calib_point(1) - offset) / sqrt(slope^2 + 1));
    %     if dist_perp > 60
    %        dist_perp = 60 + dist_perp^(1/3);
    %     end
    if num > 2
        c_dif(num+1) = sqrt((oC(1) - old_C(1))^2 + (oC(2) - old_C(2))^2);
        p_dif(num+1) = sqrt((oP(1) - old_P(1))^2 + (oP(2) - old_P(2))^2);
    end
    
    temp = sqrt((sc(1) - sp(1))^2 + (sc(2) - sp(2))^2);
    if temp > 10000
        temp = 10000;
    end
    d_sc_sp = temp;
    
    
    if num > start_num && d_sc_sp >= last_d_sc_sp_thresh
        
        %% TEST
        d(count) = abs(calib_point(1) - sc(1,1)); count = count + 1; %xdist
        d(count) = abs(calib_point(2) - sc(2,1)); count = count + 1; %ydist
        
        
    end
end
end

