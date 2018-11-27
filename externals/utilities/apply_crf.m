
function result = apply_crf_davis(rgb_img, init_mask)

    % parameter setting
    fullcrfPara.uw = 1; 
    fullcrfPara.sw = 5; 
    fullcrfPara.bw = 5;
    fullcrfPara.s = 2;  
    fullcrfPara.bl = 10; 
    fullcrfPara.bc = 10;

    numlabels = 2;

    init_label = single(init_mask(:));
    textonboost = zeros(size(rgb_img,1)*size(rgb_img,2), numlabels);
    textonboost(:,1) = init_label;
    textonboost(:,2) = 1-init_label;

    % convert matlab index to c++ index
    u = textonboost;
    u = reshape(u, size(rgb_img, 1), size(rgb_img, 2), numlabels);
    u = permute(u, [2 1 3]);
    u = reshape(u, size(rgb_img, 1)*size(rgb_img, 2), numlabels);
    u = u'*fullcrfPara.uw;

    tmpImg = reshape(rgb_img, [], 3);
    tmpImg = tmpImg';
    tmpImg = reshape(tmpImg, 3, size(rgb_img, 1), size(rgb_img, 2));
    tmpImg = permute(tmpImg, [1 3 2]);


    [L, prob] = fullCRFinfer(single(u), uint8(tmpImg), fullcrfPara.s, fullcrfPara.s, ...
        fullcrfPara.sw, fullcrfPara.bl, fullcrfPara.bl, fullcrfPara.bc, fullcrfPara.bc, ...
        fullcrfPara.bc, fullcrfPara.bw, size(rgb_img, 2), size(rgb_img, 1));

    result = (reshape(L, size(rgb_img, 2), size(rgb_img, 1)))';
    % p = reshape(prob, numlabels, size(rgb_img, 2), size(rgb_img, 1));
    % p = permute(p, [3 2 1]);

end
