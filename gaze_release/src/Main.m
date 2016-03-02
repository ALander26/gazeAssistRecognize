% Main code for gaze estimation
% Adaptive Eye-Camera Calibration for Head-Worn Devices - CVPR 2015
% Authors: David Perra, Rohit Kumar Gupta, Jan-Micheal Frahm
% please cite the above paper if you use this work
% code is free to distribute and any changes as long as the authors cite
% the above paper

clear all;clc;

if isunix
    %Linux
    addpath(genpath('./lib/'))
    addpath(genpath('../gbvs/'))
    addpath(genpath('../data/calib1/'));
else
    %Windows
    addpath(genpath('.\lib\'))
    addpath(genpath('..\gbvs\'))
    addpath(genpath('..\data\calib1\'));
end
% data =[1 2 3 5 6 7 8 9 10 11 12 13 14 15 16 17 18 21]; %database numbers
data =[1]; %database numbers
for id = 1:1
    DATASET = data(id);
    % max iterations for optimization
    ITER = [20]; %previously it was [20 50]
    winsize = [5];
    distance_to_target = 0;
    end_num = 0;
    
    if DATASET == 1
        end_num = 200;
        distance_to_target = 1050;
        width_per_px = 0.8;
    elseif DATASET == 2 %add more 
        end_num = 230;
        distance_to_target = 1050;
        width_per_px = 0.8;
    end
    clear d_sc_sp;  
    %if some data frames are not good add it to this array
    DO_NOT_USE = [];
    scene_face_info = [];
    OFFSET_X = 0;
    OFFSET_Y = 0;
    reflection_thresh = 0.8;
    pupil_thresh = 0.15;
    %details about the dataset
    if DATASET == 1
        user = 'calib_user';
        scene = 'calib_scene';
        DO_NOT_USE = [0 36 37 57 70 78 142 178 187 222 236 238];
        % load the interest points for the dataset (computed offline to
        % save time)
        load('scene_face_info.mat');
        OFFSET_X = 355;
        OFFSET_Y = 13;
        pupil_thresh = 0.15;
        
    elseif DATASET == 2
        user = 'calib_user2';
        scene = 'calib_scene2';
        DO_NOT_USE = [];%[27 98 99 101 104 105 106 107 109 110 111 112];
        load('scene_face_info2.mat');
        scene_face_info = scene_face_info2;
        OFFSET_X = 335;
        OFFSET_Y = 110;
        pupil_thresh = 0.15;        
   
    end
    
    old_C = [0; 0; 0];
    old_P = [0; 0; 0];
    last_d_sc_sp_thresh = 0;
    ln = length(ITER);
    
    %main code
    for ni = 1:ln           %test for various iteration counts (ITER)
        Niter = ITER(ni);
        for nw = 1: length(winsize)  %parfor : tested on different window sizes
            cols = (winsize(nw) + 6) * 2;
            d = zeros(cols, 1);           
            c_dif = zeros(cols, 1);
            p_dif = zeros(cols, 1);
            
            windows = zeros(floor(end_num/winsize(nw)) + 1, 16);
            start_numi = 0;
            %result = [-1.0 -3.0 -8.0 10.0 180.0 3.0 72.0 7.8 5.0 1.5];
            % initial parameters
            result = [-102.4 -28.5 -3124.6 0 0 0 3137.7 1000.8 5.0 1.5]; %3137.7
            user_eyes = cell(1,11);
            scene_eyes = cell(1,11);
            K = [ 448.63  0  340.26;  0  442.54  210.61;  0  0  1 ]; %for real
            scene_K = [ 680.678  0  313.105;  0  681.750  239.005;  0  0  1 ];
            led_positions = [23.81    0.0;
                0.0      23.81;
                0.0      0.0 ];
             reflection_thresh = 0.8;
            for start_num = start_numi : winsize(nw) : end_num-5-winsize(nw)
                tic;
                for num = start_num : start_num + winsize(nw) + 6
                    userl  = sprintf([user '%04d.png'],(num+1));
                    scenel = sprintf([scene '%04d.png'],(num+1));
                    user_eyes{num+1-start_num} = im2double(rgb2gray(imread(userl)));
                    scene_images{num+1-start_num} = im2double(rgb2gray(imread(scenel)));
                    % Find the pupil's boundary and Centroid
                    [px, py] = GetPupilBoundaryPoints(user_eyes{num+1-start_num}, pupil_thresh, reflection_thresh);
                    image_pupil_center{num+1-start_num} = fitellipse([px'; py']);
                    centroids{num+1-start_num} = FindLEDCentroids(user_eyes{num+1-start_num}, reflection_thresh, OFFSET_X, OFFSET_Y);                               
                    calib_points{num+1-start_num} = scene_face_info(num + 1, 2:3);  %ground truth for accuracy detection

                end
                user_eye = user_eyes{num+1-start_num};
                scene_image = scene_images{num+1-start_num};                
                result = OptimizeGaze(K, scene_K, led_positions, calib_points, centroids, image_pupil_center, result,0, DATASET,start_num, winsize(nw), Niter, DO_NOT_USE,scene_face_info,OFFSET_X,OFFSET_Y, user, scene, pupil_thresh, old_C, old_P,last_d_sc_sp_thresh, cols, d, c_dif, p_dif, user_eyes, scene_images );
                d = ProcessEyeFuncLinCost(K, scene_K, led_positions, calib_points, centroids, image_pupil_center, result, 6,DATASET,start_num, winsize, DO_NOT_USE,scene_face_info,OFFSET_X,OFFSET_Y, user, scene,pupil_thresh, old_C, old_P,last_d_sc_sp_thresh, cols, d, c_dif, p_dif, user_eyes,scene_images );
                windows(start_num / winsize(nw) + 1, 1) = start_num; % The starting frame for the window
                windows(start_num / winsize(nw) + 1, 2) = mean(d(1:6));  % The calibration error
                windows(start_num / winsize(nw) + 1, 3) = d(7);  % The error for the next frame
                windows(start_num / winsize(nw) + 1, 4) = mean(d(7 : 11));  % The error for 5 frames
                windows(start_num / winsize(nw) + 1, 5) = 0;%mean(d(start_num+1:start_num + 1 + 30));  % The error for 30 frames
                windows(start_num / winsize(nw) + 1, 6) = max(d(1:7));%max(d(start_num+1:start_num + 1 + 30));   % The max error for 30 frames
                
                windows(start_num / winsize(nw) + 1, 7:16) = result;
                toc;
            end
  
            windows(:,2:6) = windows(:,2:6) * width_per_px;
            windows(:,2:6) = atand(windows(:,2:6) / distance_to_target);            
            figure; plot(windows(:,2));
            M1 = 'Calibration error';
            hold on
            plot(windows(:,3), 'r');
            M2 = 'Next-frame error';
            plot(windows(:,4), 'g');
            M3 = 'Average error for 5 additional frames';

            M4 = 'Lowest mean error for Pfeuffer et al.';
            comparison = ones(1, size(windows,1)) * 0.55;
            plot(comparison, '-.black');

            M5 = 'Lowest mean error for Tan et al.';
            comparison = ones(1, size(windows,1)) * 0.38;
            plot(comparison, '-.b');

            M6 = 'Mean error for our approach';
            comparison = ones(1, size(windows,1)) * mean(windows(:,3));
            plot(comparison, '-.r');
            title(['SIMULATION ',num2str(DATASET), ': Locally Optimal Calibration Error']);
            xlabel('Calibration Set Number');
            ylabel('Error (degrees)');
            legend(M1, M2, M3, M4, M5, M6);
        end       
        
    end
end