# MAIN (ALL)

#Notlar:

# 1 . QRLSTM'ler 20 epoch yapti. bir 20 daha yapabilirler ama su an gerek yok. uzun suruyor

# 2 . Tau05 ve Tau95 fonksiyonlarini ters yapmisim. o yuzden sonuclar da ters. ama np


# LIBRARIES ---------------------------------------------------------------
require(ggplot2)
require(dplyr)
# TENSORFLOW / KERAS
library(tensorflow)
library(keras)
# install_tensorflow()
tf$constant("Hellow Tensorflow")

# other libs
require(data.table)


# Read / Prep Data ---------------------------------------------------------------

d1 = read.csv("/Users/canhakan/trafik/istanbul/istanbul/data/istanbul_data.csv")
d2 = read.csv("/Users/canhakan/trafik/istanbul/istanbul/data/istanbul_data_arima.csv")
d3 = read.csv("/Users/canhakan/trafik/istanbul/istanbul/data/istanbul_data_mean_sh.csv")
d4 = read.csv("/Users/canhakan/trafik/istanbul/istanbul/data/istanbul_data_mean_shsd.csv")
# prep data
d1 = data.table(index = c(0:(nrow(d1) - 1)),
                flow  = d1$NUMBER_OF_VEHICLES)

d2 = data.table(index = d2$X,
                flow  = d2$X0)

d3 = data.table(index = d3$X,
                flow  = d3$NUMBER_OF_VEHICLES)

d4 = data.table(index = d4$X,
                flow  = d4$NUMBER_OF_VEHICLES)


# dataya ayar -------------------------------------------------------------
# Set outliers to the corresponding quantile : 0.999 and 0.001
upquant = quantile(d1$flow,0.999) # this changes with data: d1>d2=d3=d4
downquant = quantile(d1$flow,0.001) # 12 for all datasets

d1 <- d1 %>% mutate(flow = ifelse(flow > upquant, upquant, ifelse(flow < downquant, downquant ,flow)))
d2 <- d2 %>% mutate(flow = ifelse(flow > upquant, upquant, ifelse(flow < downquant, downquant ,flow)))
d3 <- d3 %>% mutate(flow = ifelse(flow > upquant, upquant, ifelse(flow < downquant, downquant ,flow)))
d4 <- d4 %>% mutate(flow = ifelse(flow > upquant, upquant, ifelse(flow < downquant, downquant ,flow)))

# HELPER FUNCTIONS --------------------------------------------------------
createLagged <- function(data, lags, other.head = 0, other.tail = 0) {
    # remove 0 if it exists
    # get min/max for removing first/last rows for lag/lead
    minlag = min(0,lags)
    maxlag = max(0,lags)
    startN = 1 - minlag
    endN = nrow(data) - maxlag
    # seperating others
    other = c(other.head, other.tail)
    tobe.head = data[(startN:endN),..other.head]
    tobe.tail = data[(startN:endN),..other.tail]
    dat = data.table() # empty data table
    # adding lagged values one by one (column by column)
    for(i in lags){
        lagdat = data[((startN+i):(endN+i)),-..other]
        if(i<0){
            lagname = paste('lag',-i,sep='')
            colnames(lagdat) = paste(colnames(lagdat),lagname,sep='_')
        } else if(i>0){
            lagname = paste('lead',i,sep='')
            colnames(lagdat) = paste(colnames(lagdat),lagname,sep='_')
        }
        dat = cbind(dat,lagdat)
    }
    dat = cbind(tobe.head,dat,tobe.tail)
    return(dat)
}

# I create 3 pinball loss functions (for quantiles : 0.05, 0.5 and 0.95)
# i.e. one mirrors MAE others are 5th% and 95th% quantiles.
# i've learned how to just make one and change quantile as i want but i am using the previously coded function
pinball_loss50 <- function(y_pred, y_true) {
    y_pred = tf$convert_to_tensor(y_pred)
    y_true = tf$cast(y_true, y_pred$dtype)
    #
    tau = tf$expand_dims(tf$cast(0.5, y_pred$dtype), as.integer(0)) # change the 0.5 here  for other quantiles
    one = tf$cast(1, tau$dtype)
    #
    delta_y = y_true - y_pred
    pinball = tf$math$maximum(tau * delta_y, (tau - one) * delta_y)
    return(tf$reduce_mean(pinball, axis = as.integer(-1)))
}
pinball_loss05 <- function(y_pred, y_true) {
    y_pred = tf$convert_to_tensor(y_pred)
    y_true = tf$cast(y_true, y_pred$dtype)
    #
    tau = tf$expand_dims(tf$cast(0.05, y_pred$dtype), as.integer(0)) # change the 0.5 here  for other quantiles
    one = tf$cast(1, tau$dtype)
    #
    delta_y = y_true - y_pred
    pinball = tf$math$maximum(tau * delta_y, (tau - one) * delta_y)
    return(tf$reduce_mean(pinball, axis = as.integer(-1)))
}
pinball_loss95 <- function(y_pred, y_true) {
    y_pred = tf$convert_to_tensor(y_pred)
    y_true = tf$cast(y_true, y_pred$dtype)
    #
    tau = tf$expand_dims(tf$cast(0.95, y_pred$dtype), as.integer(0)) # change the 0.5 here  for other quantiles
    one = tf$cast(1, tau$dtype)
    #
    delta_y = y_true - y_pred
    pinball = tf$math$maximum(tau * delta_y, (tau - one) * delta_y)
    return(tf$reduce_mean(pinball, axis = as.integer(-1)))
}
# minmax scale: -----------------------------------------------------------
maxflow = max(d1$flow) # 489.872 (was 607 previously)
d1$flow = d1$flow / maxflow
d2$flow = d2$flow / maxflow
d3$flow = d3$flow / maxflow
d4$flow = d4$flow / maxflow


# LAGGED DATA -----------------------------------------------------------
a1 = createLagged(data = d1,lags = c(-169:-167, -25:-23, -2:0), other.head = 1)
a2 = createLagged(data = d2,lags = c(-169:-167, -25:-23, -2:0), other.head = 1)
a3 = createLagged(data = d3,lags = c(-169:-167, -25:-23, -2:0), other.head = 1)
a4 = createLagged(data = d4,lags = c(-169:-167, -25:-23, -2:0), other.head = 1)
# train/test
t1 = as.matrix(a1[1:floor(nrow(a1)*0.75), -1])
t2 = as.matrix(a2[1:floor(nrow(a2)*0.75), -1])
t3 = as.matrix(a3[1:floor(nrow(a3)*0.75), -1])
t4 = as.matrix(a4[1:floor(nrow(a4)*0.75), -1])
#
te1 = as.matrix(a1[ceiling(nrow(a1)*0.75):nrow(a1), -1])
te2 = as.matrix(a2[ceiling(nrow(a2)*0.75):nrow(a2), -1])
te3 = as.matrix(a3[ceiling(nrow(a3)*0.75):nrow(a3), -1])
te4 = as.matrix(a4[ceiling(nrow(a4)*0.75):nrow(a4), -1])
#
t1x = t1[,-9]
t2x = t2[,-9]
t3x = t3[,-9]
t4x = t4[,-9]

t1y = t1[,9]
t2y = t2[,9]
t3y = t3[,9]
t4y = t4[,9]
#
te1x = te1[,-9]
te2x = te2[,-9]
te3x = te3[,-9]
te4x = te4[,-9]

te1y = te1[,9]
te2y = te2[,9]
te3y = te3[,9]
te4y = te4[,9]
#

# data -> array -----------------------------------------------------------
# 3 boyutlu olacak : [samples, timesteps, features]
t1ax = array(data = t1x, dim = c(nrow(t1x), 9, 1))
t2ax = array(data = t2x, dim = c(nrow(t2x), 9, 1))
t3ax = array(data = t3x, dim = c(nrow(t3x), 9, 1))
t4ax = array(data = t4x, dim = c(nrow(t4x), 9, 1))
#
t1ay = array(data = t1y, dim = c(nrow(t1x), 1, 1))
t2ay = array(data = t2y, dim = c(nrow(t2x), 1, 1))
t3ay = array(data = t3y, dim = c(nrow(t3x), 1, 1))
t4ay = array(data = t4y, dim = c(nrow(t4x), 1, 1))
#
te1ax = array(data = te1x, dim = c(nrow(te1x), 9, 1))
te2ax = array(data = te2x, dim = c(nrow(te2x), 9, 1))
te3ax = array(data = te3x, dim = c(nrow(te3x), 9, 1))
te4ax = array(data = te4x, dim = c(nrow(te4x), 9, 1))
#
te1ay = array(data = te1y, dim = c(nrow(te1x), 1, 1))
te2ay = array(data = te2y, dim = c(nrow(te2x), 1, 1))
te3ay = array(data = te3y, dim = c(nrow(te3x), 1, 1))
te4ay = array(data = te4y, dim = c(nrow(te4x), 1, 1))
#


# QRLSTM ------------------------------------------------------------------

## DATA 1 ------------------------------------------------------------------
## tau = 0.05
qrlstm05_d1 <- keras_model_sequential()

qrlstm05_d1 %>%
    layer_lstm(units = 8, input_shape = c(9,1)) %>%
    layer_activation_softmax() %>%
    layer_dense(units = 1)
#
qrlstm05_d1 %>%
    compile(loss = pinball_loss05,
            optimizer = optimizer_adam(learning_rate = 0.001),
            metrics = 'mae')

fit_qr05_d1 <- qrlstm05_d1 %>% fit(
    x = t1ax,
    y = t1ay,
    batch_size = 10,
    epochs = 30,
    verbose = 1,
    shuffle = FALSE
)

## tau = 0.95
qrlstm95_d1 <- keras_model_sequential()

qrlstm95_d1 %>%
    layer_lstm(units = 8, input_shape = c(9,1)) %>%
    layer_activation_softmax() %>%
    layer_dense(units = 1)
#
qrlstm95_d1 %>%
    compile(loss = pinball_loss95,
            optimizer = optimizer_adam(learning_rate = 0.001),
            metrics = 'mae')


fit_qr95_d1 <- qrlstm95_d1 %>% fit(
    x = t1ax,
    y = t1ay,
    batch_size = 10,
    epochs = 30,
    verbose = 1,
    shuffle = FALSE
)

## DATA 2 -----------------------------------------------------------------
## tau = 0.05
qrlstm05_d2 <- keras_model_sequential()

qrlstm05_d2 %>%
    layer_lstm(units = 8, input_shape = c(9,1)) %>%
    layer_activation_softmax() %>%
    layer_dense(units = 1)
#
qrlstm05_d2 %>%
    compile(loss = pinball_loss05,
            optimizer = optimizer_adam(learning_rate = 0.001),
            metrics = 'mae')

# colab sonucu : loss = 0.008681 / mae = 0.1392 (fark = bunun learningi 10 kat daha dusuk)

fit_qr05_d2 <- qrlstm05_d2 %>% fit(
    x = t2ax,
    y = t2ay,
    batch_size = 10,
    epochs = 30,
    verbose = 1,
    shuffle = FALSE
)

## tau = 0.95
qrlstm95_d2 <- keras_model_sequential()

qrlstm95_d2 %>%
    layer_lstm(units = 8, input_shape = c(9,1)) %>%
    layer_activation_softmax() %>%
    layer_dense(units = 1)
#
qrlstm95_d2 %>%
    compile(loss = pinball_loss95,
            optimizer = optimizer_adam(learning_rate = 0.001),
            metrics = 'mae')


fit_qr95_d2 <- qrlstm95_d2 %>% fit(
    x = t2ax,
    y = t2ay,
    batch_size = 10,
    epochs = 30,
    verbose = 1,
    shuffle = FALSE
)


## DATA 3 -----------------------------------------------------------------
## tau = 0.05
qrlstm05_d3 <- keras_model_sequential()

qrlstm05_d3 %>%
    layer_lstm(units = 8, input_shape = c(9,1)) %>%
    layer_activation_softmax() %>%
    layer_dense(units = 1)
#
qrlstm05_d3 %>%
    compile(loss = pinball_loss05,
            optimizer = optimizer_adam(learning_rate = 0.001),
            metrics = 'mae')

fit_qr05_d3 <- qrlstm05_d3 %>% fit(
    x = t3ax,
    y = t3ay,
    batch_size = 10,
    epochs = 30,
    verbose = 1,
    shuffle = FALSE
)

## tau = 0.95
qrlstm95_d3 <- keras_model_sequential()

qrlstm95_d3 %>%
    layer_lstm(units = 8, input_shape = c(9,1)) %>%
    layer_activation_softmax() %>%
    layer_dense(units = 1)
#
qrlstm95_d3 %>%
    compile(loss = pinball_loss95,
            optimizer = optimizer_adam(learning_rate = 0.001),
            metrics = 'mae')


fit_qr95_d3 <- qrlstm95_d3 %>% fit(
    x = t3ax,
    y = t3ay,
    batch_size = 10,
    epochs = 30,
    verbose = 1,
    shuffle = FALSE
)

## DATA 4 -----------------------------------------------------------------
## tau = 0.05
qrlstm05_d4 <- keras_model_sequential()

qrlstm05_d4 %>%
    layer_lstm(units = 8, input_shape = c(9,1)) %>%
    layer_activation_softmax() %>%
    layer_dense(units = 1)
#
qrlstm05_d4 %>%
    compile(loss = pinball_loss05,
            optimizer = optimizer_adam(learning_rate = 0.001),
            metrics = 'mae')

fit_qr05_d4 <- qrlstm05_d4 %>% fit(
    x = t4ax,
    y = t4ay,
    batch_size = 10,
    epochs = 30,
    verbose = 1,
    shuffle = FALSE
)

## tau = 0.95
qrlstm95_d4 <- keras_model_sequential()

qrlstm95_d4 %>%
    layer_lstm(units = 8, input_shape = c(9,1)) %>%
    layer_activation_softmax() %>%
    layer_dense(units = 1)
#
qrlstm95_d4 %>%
    compile(loss = pinball_loss95,
            optimizer = optimizer_adam(learning_rate = 0.001),
            metrics = 'mae')


fit_qr95_d4 <- qrlstm95_d4 %>% fit(
    x = t4ax,
    y = t4ay,
    batch_size = 10,
    epochs = 30,
    verbose = 1,
    shuffle = FALSE
)




# all models are set i guess. they're trained enough ----------------------
# we may need to train the ones at 90s a little more. they look like they can improve
# plots
plot(ts(fit_qr05_d1$metrics$loss))
plot(ts(fit_qr05_d2$metrics$loss))
plot(ts(fit_qr05_d3$metrics$loss))
plot(ts(fit_qr05_d4$metrics$loss))
plot(ts(fit_qr95_d1$metrics$loss))
plot(ts(fit_qr95_d2$metrics$loss))
plot(ts(fit_qr95_d3$metrics$loss))
plot(ts(fit_qr95_d4$metrics$loss))
# mins
min(fit_qr05_d1$metrics$loss)
min(fit_qr05_d2$metrics$loss)
min(fit_qr05_d3$metrics$loss)
min(fit_qr05_d4$metrics$loss)
min(fit_qr95_d1$metrics$loss)
min(fit_qr95_d2$metrics$loss)
min(fit_qr95_d3$metrics$loss)
min(fit_qr95_d4$metrics$loss)




# QRLSTM - > Prediction Intervals -----------------------------------------

# Predict with QRLSTM
predtrain05_d1 <- predict(qrlstm05_d1, t1ax, batch_size = 1)
pred05_d1      <- predict(qrlstm05_d1, te1ax, batch_size = 1)
predtrain95_d1 <- predict(qrlstm95_d1, t1ax, batch_size = 1)
pred95_d1      <- predict(qrlstm95_d1, te1ax, batch_size = 1)
#
predtrain05_d2 <- predict(qrlstm05_d2, t2ax, batch_size = 1)
pred05_d2      <- predict(qrlstm05_d2, te2ax, batch_size = 1)
predtrain95_d2 <- predict(qrlstm95_d2, t2ax, batch_size = 1)
pred95_d2      <- predict(qrlstm95_d2, te2ax, batch_size = 1)
#
predtrain05_d3 <- predict(qrlstm05_d3, t3ax, batch_size = 1)
pred05_d3      <- predict(qrlstm05_d3, te3ax, batch_size = 1)
predtrain95_d3 <- predict(qrlstm95_d3, t3ax, batch_size = 1)
pred95_d3      <- predict(qrlstm95_d3, te3ax, batch_size = 1)
#
predtrain05_d4 <- predict(qrlstm05_d4, t4ax, batch_size = 1)
pred05_d4      <- predict(qrlstm05_d4, te4ax, batch_size = 1)
predtrain95_d4 <- predict(qrlstm95_d4, t4ax, batch_size = 1)
pred95_d4      <- predict(qrlstm95_d4, te4ax, batch_size = 1)

###
# Batch Size = 10
###
bpredtrain05_d1 <- predict(qrlstm05_d1, t1ax, batch_size = 10)
bpred05_d1      <- predict(qrlstm05_d1, te1ax, batch_size = 10)
bpredtrain95_d1 <- predict(qrlstm95_d1, t1ax, batch_size = 10)
bpred95_d1      <- predict(qrlstm95_d1, te1ax, batch_size = 10)
#
bpredtrain05_d2 <- predict(qrlstm05_d2, t2ax, batch_size = 10)
bpred05_d2      <- predict(qrlstm05_d2, te2ax, batch_size = 10)
bpredtrain95_d2 <- predict(qrlstm95_d2, t2ax, batch_size = 10)
bpred95_d2      <- predict(qrlstm95_d2, te2ax, batch_size = 10)
#
bpredtrain05_d3 <- predict(qrlstm05_d3, t3ax, batch_size = 10)
bpred05_d3      <- predict(qrlstm05_d3, te3ax, batch_size = 10)
bpredtrain95_d3 <- predict(qrlstm95_d3, t3ax, batch_size = 10)
bpred95_d3      <- predict(qrlstm95_d3, te3ax, batch_size = 10)
#
bpredtrain05_d4 <- predict(qrlstm05_d4, t4ax, batch_size = 10)
bpred05_d4      <- predict(qrlstm05_d4, te4ax, batch_size = 10)
bpredtrain95_d4 <- predict(qrlstm95_d4, t4ax, batch_size = 10)
bpred95_d4      <- predict(qrlstm95_d4, te4ax, batch_size = 10)



# NICE PLOTTING BEFORE SCORING --------------------------------------------
t1all = data.table(index = c(1:length(t1y)),
                   upper = predtrain05_d1,
                   flow  = t1y,
                   lower = predtrain95_d1)

te1all = data.table(index = c(1:length(te1y)),
                    upper = pred05_d1,
                    flow  = te1y,
                    lower = pred95_d1)
colnames(t1all) <- colnames(te1all) <- c("index", "upper", "flow", "lower")
#
t2all = data.table(index = c(1:length(t2y)),
                   upper = predtrain05_d2,
                   flow  = t2y,
                   lower = predtrain95_d2)

te2all = data.table(index = c(1:length(te2y)),
                    upper = pred05_d2,
                    flow  = te2y,
                    lower = pred95_d2)
colnames(t2all) <- colnames(te2all) <- c("index", "upper", "flow", "lower")
#
t3all = data.table(index = c(1:length(t3y)),
                   upper = predtrain05_d3,
                   flow  = t3y,
                   lower = predtrain95_d3)

te3all = data.table(index = c(1:length(te3y)),
                    upper = pred05_d3,
                    flow  = te3y,
                    lower = pred95_d3)
colnames(t3all) <- colnames(te3all) <- c("index", "upper", "flow", "lower")
#
t4all = data.table(index = c(1:length(t4y)),
                   upper = predtrain05_d4,
                   flow  = t4y,
                   lower = predtrain95_d4)

te4all = data.table(index = c(1:length(te4y)),
                    upper = pred05_d4,
                    flow  = te4y,
                    lower = pred95_d4)
colnames(t4all) <- colnames(te4all) <- c("index", "upper", "flow", "lower")

###
ggplot(te1all[1:300,]) +
    geom_line(mapping = aes(x = index, y = flow), color = "black") +
    geom_line(mapping = aes(x = index, y = upper), color = "blue") +
    geom_line(mapping = aes(x = index, y = lower), color = "red")




# Winkler Calculation -----------------------------------------------------


winkler(low_pred = t1all$lower, up_pred = t1all$upper, quantile = 0.9, obs = t1all$flow)
# 0.3182632 ( bir miktar daha epoch yaptirmadan once = 0.387169)
winkler(low_pred = te1all$lower, up_pred = te1all$upper, quantile = 0.9, obs = te1all$flow)
# 0.3231521 (was 0.3929869)

winkler(low_pred = t2all$lower, up_pred = t2all$upper, quantile = 0.9, obs = t2all$flow)
# 0.2177761 (was 0.2615434)
winkler(low_pred = te2all$lower, up_pred = te2all$upper, quantile = 0.9, obs = te2all$flow)
# 0.2219309 (was 0.2660739)

winkler(low_pred = t3all$lower, up_pred = t3all$upper, quantile = 0.9, obs = t3all$flow)
# 0.1957726 (was 0.2318922)
winkler(low_pred = te3all$lower, up_pred = te3all$upper, quantile = 0.9, obs = te3all$flow)
# 0.1909628 (was 0.2264859)

winkler(low_pred = t4all$lower, up_pred = t4all$upper, quantile = 0.9, obs = t4all$flow)
# 0.1928564 (was 0.2283912)
winkler(low_pred = te4all$lower, up_pred = te4all$upper, quantile = 0.9, obs = te4all$flow)
# 0.1887284 (was 0.2241348)


# Unconditional Coverage Calculation --------------------------------------

unconditional_coverage(low_pred = t1all$lower, up_pred = t1all$upper, obs = t1all$flow)
# 0.9218062 (was 0.9218062)
unconditional_coverage(low_pred = te1all$lower, up_pred = te1all$upper, obs = te1all$flow)
# 0.9144954 (was 0.9144954)

unconditional_coverage(low_pred = t2all$lower, up_pred = t2all$upper, obs = t2all$flow)
# 0.7706763 (was 0.7706763)
unconditional_coverage(low_pred = te2all$lower, up_pred = te2all$upper, obs = te2all$flow)
# 0.7710508 (was 0.7710508)

unconditional_coverage(low_pred = t3all$lower, up_pred = t3all$upper, obs = t3all$flow)
# 0.6790396 (was 0.6790396)
unconditional_coverage(low_pred = te3all$lower, up_pred = te3all$upper, obs = te3all$flow)
# 0.7146834 (was 0.7146834)

unconditional_coverage(low_pred = t4all$lower, up_pred = t4all$upper, obs = t4all$flow)
# 0.6926111 (was 0.6926111)
unconditional_coverage(low_pred = te4all$lower, up_pred = te4all$upper, obs = te4all$flow)
# 0.7174669 (was 0.7174669)






