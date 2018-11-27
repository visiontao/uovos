function result = segment_objness(objness, threshold, h, w)

    result = false(h, w);

    masks = objness.masks;
    if size(masks, 3)>1
        for i = 1:size(masks, 3)
            mask = masks(:,:,i);
            if objness.scores(i) > threshold
                result = result | mask;
            end
        end
    end
    
end