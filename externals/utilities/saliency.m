function salMot = saliency(data)

    paramMB = getParam();
    paramMB.remove_border = false; 
    paramMB.use_backgroundness = true;

    salMot = doMBS(data, paramMB);

end