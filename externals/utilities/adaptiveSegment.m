function result = adaptiveSegment(prob_map, N)

    level = multithresh(prob_map,N);
    quants = imquantize(prob_map,level);
    result = quants>1;

%     adaptive_threshold = 2.0 * mean(prob_map(:));
%     result = prob_map > adaptive_threshold;
    
    
end
