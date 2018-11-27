function warped = propagateMask(data, flow)

    vx = flow(:,:,1);
    vy = flow(:,:,2);
    
    [h,w,~] = size(flow);

    warped = zeros(h,w);
    
    for x1 = 1:w
        for y1 = 1:h
            x2 = round(x1+vx(y1,x1));
            y2 = round(y1+vy(y1,x1));
                        
            flag = isPointInView(x2,y2,w,h);
            if flag == 1
                warped(y2,x2) = data(y1,x1);
            end            
        end
    end

end

