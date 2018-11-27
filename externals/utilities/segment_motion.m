function result = segment_motion(flow)
    [h, w, ~] = size(flow);
    data = cat(3, flow, zeros(h,w));
    prob_sal = saliency(data);

    result = adaptiveSegment(prob_sal, 2);
      
end