function result = segment_motion_objness(flow, objness, threshold)
    seg_sal = segment_motion(flow);
    
    [h, w, ~] = size(flow);    
    result = false(h, w);
    
    masks = objness.masks;
    if size(masks, 3)>1
        for i = 1:size(masks, 3)
            mask = masks(:,:,i);
            delta = iou(seg_sal, mask);
            if delta > threshold
                result = result | mask;
            end
        end
    end
    
    result = result | seg_sal;
end