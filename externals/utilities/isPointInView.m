function flag = isPointInView(x,y,w,h)
    flag = 1;
    if x<1 || x>w || y<1 || y>h
        flag = 0;
    end
end