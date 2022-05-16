# LSTM Normal (for Forecast Similarity)

# not: LSTM modellerinin epoch sayisi total 40-50 civari. daha da fazla yapsam daha da iyilesir aslinda

# A Different Partition -------------------------------
# train/test
tt1 = as.matrix(a1[1:floor(nrow(a1)*0.5), -1])
tt2 = as.matrix(a2[1:floor(nrow(a2)*0.5), -1])
tt3 = as.matrix(a3[1:floor(nrow(a3)*0.5), -1])
tt4 = as.matrix(a4[1:floor(nrow(a4)*0.5), -1])
#
tte1 = as.matrix(a1[ceiling(nrow(a1)*0.5):nrow(a1), -1])
tte2 = as.matrix(a2[ceiling(nrow(a2)*0.5):nrow(a2), -1])
tte3 = as.matrix(a3[ceiling(nrow(a3)*0.5):nrow(a3), -1])
tte4 = as.matrix(a4[ceiling(nrow(a4)*0.5):nrow(a4), -1])
#
tt1x = t1[,-9]
tt2x = t2[,-9]
tt3x = t3[,-9]
tt4x = t4[,-9]

tt1y = t1[,9]
tt2y = t2[,9]
tt3y = t3[,9]
tt4y = t4[,9]
#
tte1x = te1[,-9]
tte2x = te2[,-9]
tte3x = te3[,-9]
tte4x = te4[,-9]

tte1y = te1[,9]
tte2y = te2[,9]
tte3y = te3[,9]
tte4y = te4[,9]
#

# data -> array
# 3 boyutlu olacak : [samples, timesteps, features]
tt1ax = array(data = tt1x, dim = c(nrow(tt1x), 9, 1))
tt2ax = array(data = tt2x, dim = c(nrow(tt2x), 9, 1))
tt3ax = array(data = tt3x, dim = c(nrow(tt3x), 9, 1))
tt4ax = array(data = tt4x, dim = c(nrow(tt4x), 9, 1))
#
tt1ay = array(data = tt1y, dim = c(nrow(tt1x), 1, 1))
tt2ay = array(data = tt2y, dim = c(nrow(tt2x), 1, 1))
tt3ay = array(data = tt3y, dim = c(nrow(tt3x), 1, 1))
tt4ay = array(data = tt4y, dim = c(nrow(tt4x), 1, 1))
#
tte1ax = array(data = tte1x, dim = c(nrow(tte1x), 9, 1))
tte2ax = array(data = tte2x, dim = c(nrow(tte2x), 9, 1))
tte3ax = array(data = tte3x, dim = c(nrow(tte3x), 9, 1))
tte4ax = array(data = tte4x, dim = c(nrow(tte4x), 9, 1))
#
tte1ay = array(data = tte1y, dim = c(nrow(tte1x), 1, 1))
tte2ay = array(data = tte2y, dim = c(nrow(tte2x), 1, 1))
tte3ay = array(data = tte3y, dim = c(nrow(tte3x), 1, 1))
tte4ay = array(data = tte4y, dim = c(nrow(tte4x), 1, 1))
#

# LSTM --------------------------------------------------------------------
## DATA 1 -------------------------------------------------------------
lstm_d1 <- keras_model_sequential()

lstm_d1 %>%
    layer_lstm(units = 8, input_shape = c(9,1)) %>%
    layer_activation_softmax() %>%
    layer_dense(units = 1)
#
lstm_d1 %>%
    compile(loss = "mae",
            optimizer = optimizer_adam(learning_rate = 0.001),
            metrics = "mae")

fit_lstm_d1 <- lstm_d1 %>% fit(
    x = tt1ax,
    y = tt1ay,
    batch_size = 1,
    epochs = 30,
    verbose = 1,
    shuffle = FALSE
)



## DATA 2 -------------------------------------------------------------
lstm_d2 <- keras_model_sequential()

lstm_d2 %>%
    layer_lstm(units = 8, input_shape = c(9,1)) %>%
    layer_activation_softmax() %>%
    layer_dense(units = 1)
#
lstm_d2 %>%
    compile(loss = "mae",
            optimizer = optimizer_adam(learning_rate = 0.001),
            metrics = "mae")

fit_lstm_d2 <- lstm_d2 %>% fit(
    x = tt2ax,
    y = tt2ay,
    batch_size = 1,
    epochs = 30,
    verbose = 1,
    shuffle = FALSE
)

## DATA 3 -------------------------------------------------------------
lstm_d3 <- keras_model_sequential()

lstm_d3 %>%
    layer_lstm(units = 8, input_shape = c(9,1)) %>%
    layer_activation_softmax() %>%
    layer_dense(units = 1)
#
lstm_d3 %>%
    compile(loss = "mae",
            optimizer = optimizer_adam(learning_rate = 0.001),
            metrics = "mae")

fit_lstm_d3 <- lstm_d3 %>% fit(
    x = tt3ax,
    y = tt3ay,
    batch_size = 1,
    epochs = 30,
    verbose = 1,
    shuffle = FALSE
)

## DATA 4 -------------------------------------------------------------
lstm_d4 <- keras_model_sequential()

lstm_d4 %>%
    layer_lstm(units = 8, input_shape = c(9,1)) %>%
    layer_activation_softmax() %>%
    layer_dense(units = 1)
#
lstm_d4 %>%
    compile(loss = "mae",
            optimizer = optimizer_adam(learning_rate = 0.001),
            metrics = "mae")

fit_lstm_d4 <- lstm_d4 %>% fit(
    x = tt4ax,
    y = tt4ay,
    batch_size = 1,
    epochs = 30,
    verbose = 1,
    shuffle = FALSE
)

# All Models are Fit.  See plots ------------------------------------------

# plots
plot(ts(fit_lstm_d1$metrics$loss))
plot(ts(fit_lstm_d2$metrics$loss))
plot(ts(fit_lstm_d3$metrics$loss))
plot(ts(fit_lstm_d4$metrics$loss))
# mins
min(fit_lstm_d1$metrics$loss) * maxflow # 28.18192 # or tail(f$m$loss , 1) ayni sonuç
min(fit_lstm_d2$metrics$loss) * maxflow # 25.42235 # or tail(... , 1) ayni sonuç
min(fit_lstm_d3$metrics$loss) * maxflow # 25.70138 # or tail(... , 1) ayni sonuç
min(fit_lstm_d4$metrics$loss) * maxflow # 25.55927 # or tail(... , 1) ayni sonuç

# LSTM - > Forecast Results -----------------------------------------

# Predict with LSTM

predtrain_d1 <- predict(lstm_d1, tt1ax, batch_size = 1)
pred_d1      <- predict(lstm_d1, tte1ax, batch_size = 1)
#
predtrain_d2 <- predict(lstm_d2, tt2ax, batch_size = 1)
pred_d2      <- predict(lstm_d2, tte2ax, batch_size = 1)
#
predtrain_d3 <- predict(lstm_d3, tt3ax, batch_size = 1)
pred_d3      <- predict(lstm_d3, tte3ax, batch_size = 1)
#
predtrain_d4 <- predict(lstm_d4, tt4ax, batch_size = 1)
pred_d4      <- predict(lstm_d4, tte4ax, batch_size = 1)



# NICE PLOTTING BEFORE SCORING --------------------------------------------
t1lstm = data.table(index = c(1:length(tt1y)),
                   flow  = tt1y,
                   pred = predtrain_d1)

te1lstm = data.table(index = c((length(tt1y) + 1):(length(tt1y) + length(tte1y))),
                    flow  = tte1y,
                    pred = pred_d1)
colnames(t1lstm) <- colnames(te1lstm) <- c("index", "flow", "pred")
#
t2lstm = data.table(index = c(1:length(tt2y)),
                   flow  = tt2y,
                   lower = predtrain_d2)

te2lstm = data.table(index = c((length(tt2y) + 1):(length(tt2y) + length(tte2y))),
                    flow  = tte2y,
                    lower = pred_d2)
colnames(t2lstm) <- colnames(te2lstm) <- c("index", "flow", "pred")
#
t3lstm = data.table(index = c(1:length(tt3y)),
                   flow  = tt3y,
                   lower = predtrain_d3)

te3lstm = data.table(index = c((length(tt3y) + 1):(length(tt3y) + length(tte3y))),
                    flow  = tte3y,
                    lower = pred_d3)
colnames(t3lstm) <- colnames(te3lstm) <- c("index", "flow", "pred")
#
t4lstm = data.table(index = c(1:length(tt4y)),
                   flow  = tt4y,
                   lower = predtrain_d4)

te4lstm = data.table(index = c((length(tt4y) + 1):(length(tt4y) + length(tte4y))),
                    flow  = tte4y,
                    lower = pred_d4)
colnames(t4lstm) <- colnames(te4lstm) <- c("index", "flow", "pred")

###
ggplot(te3lstm[1:300,]) +
    geom_line(mapping = aes(x = index, y = flow), color = "black") +
    geom_line(mapping = aes(x = index, y = pred), color = "blue")



# MAE Scores: -------------------------------------------------------------

# mins
min(fit_lstm_d1$metrics$loss) * maxflow # 28.64406
min(fit_lstm_d2$metrics$loss) * maxflow # 26.01578
min(fit_lstm_d3$metrics$loss) * maxflow # 26.03706
min(fit_lstm_d4$metrics$loss) * maxflow # 26.14599

# predicted on train (should be same as above) (BUT IS NOTTT)
mean(abs(t1lstm$flow - predtrain_d1)) * maxflow # 28.74575
mean(abs(t2lstm$flow - predtrain_d2)) * maxflow # 27.94562
mean(abs(t3lstm$flow - predtrain_d3)) * maxflow # 29.0444
mean(abs(t4lstm$flow - predtrain_d4)) * maxflow # 26.28531
# test mae
mean(abs(te1lstm$flow - pred_d1)) * maxflow # 32.26083
mean(abs(te2lstm$flow - pred_d2)) * maxflow # 28.47678
mean(abs(te3lstm$flow - pred_d3)) * maxflow # 27.78695
mean(abs(te4lstm$flow - pred_d4)) * maxflow # 24.08118



