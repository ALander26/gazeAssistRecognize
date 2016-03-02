function [ P ] = FindPupilCenter(C, rj, distance_from_C_to_P, n1, alpha_eye, beta_eye)
transformed_C = C - C;
transformed_rj = rj - C;
transformed_O = -C;

new_x_hat = (-transformed_O);
new_x_hat = new_x_hat / norm(new_x_hat);
new_z_hat = cross(new_x_hat, (transformed_O - transformed_rj));
new_z_hat = new_z_hat / norm(new_z_hat);
new_y_hat = cross(new_x_hat, new_z_hat);
new_y_hat = new_y_hat / norm(new_y_hat);
rotation_matrix = [new_x_hat, new_y_hat, new_z_hat];


transformed_C = rotation_matrix' * transformed_C;
transformed_rj = rotation_matrix' * transformed_rj;
transformed_O = rotation_matrix' * transformed_O;

r = distance_from_C_to_P;
% for theta = 0 : 0.01 : 3.14
%     pt = [r * cos(theta); r * sin(theta); 0];
%     lhs = n1 * norm(cross((transformed_rj - transformed_C), (pt - transformed_rj))) * norm(transformed_O - transformed_rj);
%     rhs = norm(cross((transformed_rj - transformed_C), (transformed_O - transformed_rj))) * norm(pt - transformed_rj);
%     error = norm(rhs - lhs);
%     if error < best_error
%        best_error = error;
%        best_pt = pt;
%        best_theta = theta;
%     end
% end
% best_pt

%theta = 0 : 0.00001 : pi/2;
%theta = 0 : pi/2;
theta = 0 : 0.01 : 2*pi;
len = size(theta,2);
pt = [r * cos(theta); r * sin(theta); zeros(1,len)];
temp = (repmat(transformed_rj - transformed_C, 1, len));
cross_prod = cross(temp, (pt - repmat(transformed_rj, 1, len)));
lhs = n1 * sqrt(sum(cross_prod.^2, 1)) * norm(transformed_O - transformed_rj);
cross_prod = cross(temp, repmat(transformed_O - transformed_rj, 1, len));
rhs_temp = pt - repmat(transformed_rj, 1, len);
rhs = sqrt(sum(cross_prod.^2, 1)) .* sqrt(sum(rhs_temp.^2, 1));
error = abs(rhs-lhs);
% d_pt_to_rj = (pt - repmat(transformed_rj, 1, size(theta, 2))).^2;
% d_pt_to_rj = sqrt(sum(d_pt_to_rj, 1));
%figure; plot(d_pt_to_o);
%figure; plot(error);

%error = 1000 * d_pt_to_rj + error;
%figure; plot(error);
error = error - pt(2,:) * 300;
%hold on; plot(error, 'Color', 'red');
best_index = find(error == min(error), 1);
%     best_error = min(error)
%     best_theta = theta(best_index)
P_optic = pt(:, best_index); %best_pt

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%% Adjust pupil to accomodate visual axis
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
vec = (P_optic) / norm(P_optic);
phi_eye = asind(vec(2));
theta_eye = asind(vec(1) / cosd(phi_eye));
%kg = (transformed_C(3)) / (cosd(phi_eye + beta_eye)*cosd(theta_eye + alpha_eye));
% display(phi_eye)
% display(beta_eye)
% display(theta_eye)
% display(alpha_eye)
gaze_adjustment_vec = [(cosd(real(phi_eye + beta_eye))*sind(real(theta_eye + alpha_eye)));
    sind(real(phi_eye + beta_eye));
    (-cosd(real(phi_eye + beta_eye))*cosd(real(theta_eye + alpha_eye))) ];
% P_visual = kg * gaze_adjustment_vec;
P_visual = distance_from_C_to_P * gaze_adjustment_vec;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Finally, solve for the 3d pupil center
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% P_visual(1) = P_visual(1) * -1;
% P_visual(2) = P_visual(2) * -1;

P = (rotation_matrix' \ P_visual);
P = P + C;
% P = P / norm(P);
% P = P * ((C(3) - 4.2) / P(3));

end

