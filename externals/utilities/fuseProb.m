function result = fuseProb(prob_curr, prob_hist, factor)
    result = factor*prob_curr+(1-factor)*prob_hist;
    result = imgaussfilt(result,1);

end