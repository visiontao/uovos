
function delta = iou(mask1, mask2)

    m1 = logical(mask1(:));
    m2 = logical(mask2(:));    

    d1 = sum(m1 & m2);
    d2 = sum(m1)+sum(m2)-d1;
        
    if d2==0
        delta=1;
    else
        delta = d1/d2;    
    end
    
end

% function [J, inters, fp, fn] = iou( object, ground_truth )
% 
% % Make sure they're binary
% object       = logical(object);
% ground_truth = logical(ground_truth);
% 
% % Intersection between all sets
% inters = object.*ground_truth;
% fp     = object.*(1-inters);
% fn     = ground_truth.*(1-inters);
% 
% % Areas of the intersections
% inters = sum(inters(:)); % Intersection
% fp     = sum(fp(:)); % False positives
% fn     = sum(fn(:)); % False negatives
% 
% % Compute the fraction
% denom = inters + fp + fn;
% if denom==0
%     J = 1;
% else
%     J =  inters/denom;
% end
