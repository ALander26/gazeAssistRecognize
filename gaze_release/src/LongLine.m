function [m, c] = LongLine(p1,p2, is_shifted, draw_line, image_width, image_height)

    m = (p2(2) - p1(2)) / (p2(1) - p1(1));
    c = p1(2) - m * p1(1);  
   
end