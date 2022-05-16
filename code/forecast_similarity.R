# historical forecast similarity
require(knn.covertree)
require(Rfast)

# NOT: ONEMLI: 30 epoch'tan sonra 10-15 epoch daha yaptim
# ve butun LSTMlerin MAE'si iyi bir miktar daha dustu.
# AMAAA historical prediction icin neredeyse hicbir sey degismedi

# combine all forecasts ---------------------------------------------------
lstm1 = rbind(t1lstm,te1lstm)
lstm2 = rbind(t2lstm,te2lstm)
lstm3 = rbind(t3lstm,te3lstm)
lstm4 = rbind(t4lstm,te4lstm)
#


# let's try 3 different similarity techniques:

# 1. lag10 eucliden distance => 20 closest
# 2. lag3  euclidean distance => 20 closest
# 3. lag3 => 50 => lag10 => 20
# 4. lag3 => 50 => lag10 correlation => 20


# (lag10, euclidean, 20) ------------------------------------------

## DATA 1 ----------------------------------------------------------------
# Creating Time Windows for distance calculation
lstm1lag10 = createLagged(lstm1, c(-10:0), other.head = 1, other.tail = 3)
# find_knn (set 100 then we will choose the best 20 from the ones that are from past)
ctemp_d1 = find_knn(data = lstm1lag10[ ,-c(1, 12:13)],
                    k = 100,
                    query = lstm1lag10[-c(1:nrow(t1lstm)), -c(1, 12:13)]) # takes seconds
ctemp_d1 = ctemp_d1$index
ctemp_d1 = ctemp_d1 + 10
ctempp_d1 = cbind(ctemp_d1, c((nrow(t1lstm) + 11):nrow(lstm1)))

# function for filtering out future similars
custocc = function(rowx, nr){
    return(rowx < nr)
}
# temp is a boolean matrix (false => future / true => past)
temp = apply(ctempp_d1[,-101],2,custocc,nr = ctempp_d1[,101])

sum(apply(temp,1,sum) < 20) # we have 0 time indices with less than 20 closest from past. we will repeat other closests at that time

# boolean matrix to list of indices (temp => tempL)
tempL = apply(temp,1,which)

# sorted_d1 doesnt actually sort the neighbors but will keep the closest 20 neighbors that could pass the filter
sorted_d1 = matrix(0, nrow = dim(ctempp_d1)[1], ncol = 20)

for (i in c(1:nrow(ctempp_d1))) {
    sorted_d1[i,] = ctempp_d1[i,][tempL[[i]]][1:20]
}
which(is.na(sorted_d1)) # NO

# d1_pilar = d1_PI'lar :: i.e. contains flow value of neighbors. will be used for PI estimation
d1_pilar = as.matrix(cbind(te1lstm[-c(1:10),], matrix(0, ncol = 20, nrow = nrow(te1lstm) - 10)))

for (i in c(1:nrow(d1_pilar))) {
    d1_pilar[i,4:23] = lstm1[sorted_d1[i,],]$flow
}


# quick check for MAE
tmpa = rowmeans(d1_pilar[,-c(1:3)])

mean(abs(d1_pilar[,2] - tmpa)) * maxflow # 31.59523


### PI ----------------------------------------------------------------------
# here we sort the similar time's values for creating PI (2nd neigh estimates %5th quantile)
tempS = rowSort(d1_pilar[,-c(1:3)])

qua_d1 = cbind(d1_pilar[,2], tempS[,c(2,19)])

colnames(qua_d1) = c("flow", "lower", "upper")

qua_d1 = as.data.frame(qua_d1)

### Winkler Score ----------------------------------------------------

winkler(low_pred = qua_d1$lower, up_pred = qua_d1$upper, quantile = 0.9, obs = qua_d1$flow)
# 0.2419179  (LSTM was = 0.387169(train) / 0.3929869(test))

### Unconditional Coverage Calculation --------------------------------------

unconditional_coverage(low_pred = qua_d1$lower, up_pred = qua_d1$upper, obs = qua_d1$flow)
# 0.8235727  (LSTM was = 0.9218062(train) / 0.9144954(test))



## DATA 2 ----------------------------------------------------------------
lstm2lag10 = createLagged(lstm2, c(-10:0), other.head = 1, other.tail = 3)
ctemp_d2 = find_knn(data = lstm2lag10[ ,-c(1, 12:13)],
                    k = 100,
                    query = lstm2lag10[-c(1:nrow(t2lstm)), -c(1, 12:13)]) # takes seconds
ctemp_d2 = ctemp_d2$index
ctemp_d2 = ctemp_d2 + 10
ctempp_d2 = cbind(ctemp_d2, c((nrow(t2lstm) + 11):nrow(lstm2)))

temp = apply(ctempp_d2[,-101], 2, custocc, nr = ctempp_d2[,101])

sum(apply(temp,1,sum) < 20) # we have 63 time indices with less than 20 closest from past. we will repeat other closests at that time

tempL = apply(temp,1,which)

sorted_d2 = matrix(0, nrow = dim(ctempp_d2)[1], ncol = 20)

for (i in c(1:nrow(ctempp_d2))) {
    sorted_d2[i,] = ctempp_d2[i,][tempL[[i]]][1:20]
}

# reuse other neighbours in the rows with NAs
for (i in which(rowSums(is.na(sorted_d2)) > 0)) {
    sorted_d2[i,] = rep(na.omit(sorted_d2[i,]),10)[1:20]
}


d2_pilar = as.matrix(cbind(te2lstm[-c(1:10),], matrix(0, ncol = 20, nrow = nrow(te2lstm)-10)))

for (i in c(1:nrow(d2_pilar))) {
    d2_pilar[i,4:23] = lstm2[sorted_d2[i,],]$flow
}

tmpa = apply(d2_pilar[,-c(1:3)],1,mean)

mean(abs(d2_pilar[,2] - tmpa)) * maxflow # 28.57183



### PI ----------------------------------------------------------------------
tempS = rowSort(d2_pilar[,-c(1:3)])

qua_d2 = as.data.frame(cbind(d2_pilar[,2], tempS[,c(2,19)]))

colnames(qua_d2) = c("flow", "lower", "upper")

### Winkler Score ----------------------------------------------------

winkler(low_pred = qua_d2$lower, up_pred = qua_d2$upper, quantile = 0.9, obs = qua_d2$flow)
# 0.2218581  (LSTM was = 0.2615434(train) / 0.2660739(test))

### Unconditional Coverage Calculation --------------------------------------

unconditional_coverage(low_pred = qua_d2$lower, up_pred = qua_d2$upper, obs = qua_d2$flow)
# 0.7503492  (LSTM was = 0.7706763(train) / 0.7710508(test))



## DATA 3 ----------------------------------------------------------------
lstm3lag10 = createLagged(lstm3, c(-10:0), other.head = 1, other.tail = 3)
ctemp_d3 = find_knn(data = lstm3lag10[ ,-c(1, 12:13)],
                    k = 100,
                    query = lstm3lag10[-c(1:nrow(t3lstm)), -c(1, 12:13)]) # takes seconds
ctemp_d3 = ctemp_d3$index
ctemp_d3 = ctemp_d3 + 10
ctempp_d3 = cbind(ctemp_d3, c((nrow(t3lstm) + 11):nrow(lstm3)))

temp = apply(ctempp_d3[,-101], 2, custocc, nr = ctempp_d3[,101])

sum(apply(temp,1,sum) < 20) # we have 3 time indices with less than 20 closest from past. we will repeat other closests at that time

tempL = apply(temp,1,which)

sorted_d3 = matrix(0, nrow = dim(ctempp_d3)[1], ncol = 20)

for (i in c(1:nrow(ctempp_d3))) {
    sorted_d3[i,] = ctempp_d3[i,][tempL[[i]]][1:20]
}

# reuse other neighbours in the rows with NAs
for (i in which(rowSums(is.na(sorted_d3)) > 0)) {
    sorted_d3[i,] = rep(na.omit(sorted_d3[i,]),10)[1:20]
}

d3_pilar = as.matrix(cbind(te3lstm[-c(1:10),], matrix(0, ncol = 20, nrow = nrow(te3lstm)-10)))

for (i in c(1:nrow(d3_pilar))) {
    d3_pilar[i,4:23] = lstm3[sorted_d3[i,],]$flow
}

tmpa = apply(d3_pilar[,-c(1:3)],1,mean)

mean(abs(d3_pilar[,2] - tmpa)) * maxflow # 28.59466



### PI ----------------------------------------------------------------------
tempS = rowSort(d3_pilar[,-c(1:3)])

qua_d3 = as.data.frame(cbind(d3_pilar[,2], tempS[,c(2,19)]))

colnames(qua_d3) = c("flow", "lower", "upper")

# Winkler Score ----------------------------------------------------

winkler(low_pred = qua_d3$lower, up_pred = qua_d3$upper, quantile = 0.9, obs = qua_d3$flow)
# 0.2319301  (LSTM was = 0.2318922(train) / 0.2264859(test))

# Unconditional Coverage Calculation --------------------------------------

unconditional_coverage(low_pred = qua_d3$lower, up_pred = qua_d3$upper, obs = qua_d3$flow)
# 0.8365922  (LSTM was = 0.6790396(train) / 0.7146834(test))



## DATA 4 ----------------------------------------------------------------
lstm4lag10 = createLagged(lstm4, c(-10:0), other.head = 1, other.tail = 3)
ctemp_d4 = find_knn(data = lstm4lag10[ ,-c(1, 12:13)],
                    k = 100,
                    query = lstm4lag10[-c(1:nrow(t4lstm)), -c(1, 12:13)]) # takes seconds
ctemp_d4 = ctemp_d4$index
ctemp_d4 = ctemp_d4 + 10
ctempp_d4 = cbind(ctemp_d4, c((nrow(t4lstm) + 11):nrow(lstm4)))

temp = apply(ctempp_d4[,-101], 2, custocc, nr = ctempp_d4[,101])

sum(apply(temp,1,sum) < 20) # we have 0 time indices with less than 20 closest from past. we will repeat other closests at that time

tempL = apply(temp,1,which)

sorted_d4 = matrix(0, nrow = dim(ctempp_d4)[1], ncol = 20)

for (i in c(1:nrow(ctempp_d4))) {
    sorted_d4[i,] = ctempp_d4[i,][tempL[[i]]][1:20]
}


d4_pilar = as.matrix(cbind(te4lstm[-c(1:10),], matrix(0, ncol = 20, nrow = nrow(te4lstm)-10)))

for (i in c(1:nrow(d4_pilar))) {
    d4_pilar[i,4:23] = lstm4[sorted_d4[i,],]$flow
}

tmpa = apply(d4_pilar[,-c(1:3)],1,mean)

mean(abs(d4_pilar[,2] - tmpa)) * maxflow # 28.85006


### PI ----------------------------------------------------------------------
tempS = rowSort(d4_pilar[,-c(1:3)])

qua_d4 = as.data.frame(cbind(d4_pilar[,2], tempS[,c(2,19)]))

colnames(qua_d4) = c("flow", "lower", "upper")

# Winkler Score ----------------------------------------------------

winkler(low_pred = qua_d4$lower, up_pred = qua_d4$upper, quantile = 0.9, obs = qua_d4$flow)
# 0.2305406  (LSTM was = 0.2283912(train) / 0.2241348(test))

# Unconditional Coverage Calculation --------------------------------------

unconditional_coverage(low_pred = qua_d4$lower, up_pred = qua_d4$upper, obs = qua_d4$flow)
# 0.8400838  (LSTM was = 0.6926111(train) / 0.7174669(test))
