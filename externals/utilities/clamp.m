function y = clamp(x, lb, ub)
% Clamp the value using lowerBound and upperBound

    y = max(x, lb);
    y = min(y, ub);

end