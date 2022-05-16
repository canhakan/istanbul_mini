# Winkler Score:
# for a specific quantile
# [for the whole distrubition just take the mean of all quantile winkler scores]
# pred should be renamed as real

winkler <- function(low_pred, up_pred, quantile, obs){
    # split
    higs = (obs > up_pred)
    lows = (obs < low_pred)
    obs_up = obs[higs]
    obs_low = obs[lows]
    # print(length(obs_up))
    # print(length(up_pred[higs]))
    # print(length(obs_low))
    # print(length(low_pred[lows]))
    if(length(obs_low) != length(obs_low[lows])){
        print(quantile)
        print(length(obs_low))
        print(length(low_pred[lows]))
        print('---')
    }
    # step function steps
    score_up  = sum((obs_up - (up_pred[higs]))*2/quantile)
    score_low = sum((low_pred[lows] - (obs_low))*2/quantile)
    score_all = sum(up_pred - low_pred) + score_up + score_low
    score_mean = score_all / length(obs)
    return(score_mean)
}


# Unconditional Coverage ----------------------------------------------------------
unconditional_coverage <- function(low_pred, up_pred, obs){
    # count(t1all %>% filter(lower > flow | upper < flow))
    # obs = data.table(obs)
    # oin = obs[obs>low_pred & obs<up_pred]
    # return(length(oin)/length(obs))
    n1 = sum(obs > low_pred & obs < up_pred)
    n2 = length(obs)
    return(n1/n2)
}

