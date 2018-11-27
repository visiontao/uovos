function flow = opticalFlow_siftFlow(frame1, frame2)


    alpha = 0.012;
    ratio = 0.75;
    minWidth = 20;
    nOuterFPIterations = 7;
    nInnerFPIterations = 1;
    nSORIterations = 30;

    para = [alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nSORIterations];

    [vx,vy,~] = Coarse2FineTwoFrames(im2double(frame1), im2double(frame2),para);

    flow(:,:,1) = vx;
    flow(:,:,2) = vy;

end
