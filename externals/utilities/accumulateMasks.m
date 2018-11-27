function result = accumulateMasks(mask_list, flow_list)

    [h, w, ~] = size(mask_list{1,1});
    mask_propgate = zeros(h, w);
    for i = 1:numel(mask_list)
        forewardFlow = -flow_list{i,1};

        mask = imdilate(mask_list{i,1}, strel('disk', 3));            
        
        mask_propgate = mask_propgate+double(mask);
        mask_propgate = propagateMask(mask_propgate, forewardFlow);
    end        
    result = mask_propgate/(numel(flow_list));     

end