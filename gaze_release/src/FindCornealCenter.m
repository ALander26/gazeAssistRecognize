function [C] = FindCornealCenter( led_positions, reflection_world, corneal_radius, distance_to_eye)
b1 = cross(led_positions(:, 1), reflection_world(:, 1));
b2 = cross(led_positions(:, 2), reflection_world(:, 2));
b = cross(b1, b2);
b_norm = b / norm(b);

C = (corneal_radius + distance_to_eye) * b_norm;


end

