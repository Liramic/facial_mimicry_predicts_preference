# LIBRARIES
library(rwa)
library(ggplot2)
library(dplyr)
library(lme4)

# removing outlayers:
# --- LOAD DATA ---
graphs_dir = "//"
data <- read.csv("reading.csv")
data$listener_choice <- as.factor(data$listener_choice)


null_model <- glmer(listener_choice ~ 1 + (1 | story_1), data = data, family = "binomial")
summary(null_model)

library(performance)
performance::icc(null_model) # icc is 0.055 --> no need to use mixed models.


# change the type to compute paired t-tests and permutation test
# to test mimicry use "corr"
# to test listener mean activation use "mean_comp_listener".
type <- "mean_comp_listener"

#___________________________________________________
#____________t-test for each cluster________________
#___________________________________________________
n_clusters <- 16
t_values <- numeric(n_clusters)
p_values <- numeric(n_clusters)
dfs <- numeric(n_clusters)
sig_labels <- character(n_clusters)
cluster_names <- paste0("Cluster ", 1:(n_clusters))

# Loop through each cluster and perform paired t-test
for (i in 0:(n_clusters - 1)) {
  # Get both conditions
  S1 <- data[[paste0("cluster_", i, "_", type, "_S1")]]
  S2 <- data[[paste0("cluster_", i, "_", type, "_S2")]]
  
  # Compute "chosen" and "unchosen" values
  chosen <- ifelse(data$listener_choice == 0, S1, S2)
  unchosen <- ifelse(data$listener_choice == 0, S2, S1)

  # Paired t-test
  test_result <- t.test(chosen, unchosen, paired = TRUE)
  t_values[i + 1] <- test_result$statistic
  p_values[i + 1] <- test_result$p.value
  dfs[i+1] <- test_result$parameter
  
  # Add significance stars
  if (p_values[i + 1] < 0.001) {
    sig_labels[i + 1] <- "***"
  } else if (p_values[i + 1] < 0.01) {
    sig_labels[i + 1] <- "**"
  } else if (p_values[i + 1] < 0.05) {
    sig_labels[i + 1] <- "*"
  } else {
    sig_labels[i + 1] <- ""
  }
}


# Data frame for plotting
t_df <- data.frame(
  cluster = factor(cluster_names, levels = rev(cluster_names)),
  t_value = t_values,
  p_value = p_values,
  df = dfs,
  sig = sig_labels
)

ggplot(t_df, aes(x = t_value, y = cluster)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_vline(xintercept = 0, linetype = "dotted", color = "black") +
  geom_text(aes(label = paste0(round(t_value, 2), sig)),
            hjust = ifelse(t_df$t_value > 0, -0.2, 1.2),
            size = 5) +
  theme_minimal() +
  theme(
    axis.text.y = element_text(size = 14),
    axis.text.x = element_text(size = 18),
    axis.title.y = element_text(size = 16),
    axis.title.x = element_text(size = 16),
    plot.title = element_text(size = 18, face = "bold")
  ) +
  labs(
    title = "",
    x = "",
    y = "Cluster"
  ) +
  #coord_cartesian(xlim=c(-0.5, 2.5))
  coord_cartesian(xlim = c(-2.2, 3.5))

print(t_df)

ggsave(paste0(graphs_dir, "t_test_cluster_plot_", type, ".png"), width = 6, height = 6, dpi = 300, bg = 'white')


#___________________________________________________
#____________permutaion prefrence based models______
#___________________________________________________


# --- SAFE Z-SCORE ---
safe_zscore <- function(x) {
  (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
}


build_weighted_predictors <- function(df, outcome = "listener_choice", predictors, print=FALSE) {
  # Run RWA
  rwa_out <- rwa(df = df, outcome = outcome, predictors = predictors, applysigns = TRUE)
  
  # Parse sign and weights
  rwa_df <- rwa_out$result
  
  pos_vars <- rwa_df$Variables[rwa_df$Sign == "+"]
  neg_vars <- rwa_df$Variables[rwa_df$Sign == "-"]
  
  pos_wts <- abs(rwa_df$Sign.Rescaled.RelWeight[rwa_df$Sign == "+"])
  neg_wts <- abs(rwa_df$Sign.Rescaled.RelWeight[rwa_df$Sign == "-"])
  
  if (print) {
    cat("Positive variables:\n")
    cat(pos_vars)
    cat("\n")
    cat("Positive weights:\n")
    cat(pos_wts)
    cat("\n")
    cat("Negative variables:\n")
    cat(neg_vars)
    cat("\n")
    cat("Negative weights:\n")
    cat(neg_wts)
    cat("\n")
  }
  
  # Weighted sums
  df$pos <- 0
  if (length(pos_vars) > 0) {
    pos_mat <- df[, pos_vars, drop = FALSE]
    for (j in seq_along(pos_vars)) {
      pos_mat[, j] <- pos_mat[, j] * pos_wts[j]
    }
    df$pos <- rowSums(pos_mat, na.rm = TRUE) / sum(pos_wts)
  }
  
  df$neg <- 0
  if (length(neg_vars) > 0) {
    neg_mat <- df[, neg_vars, drop = FALSE]
    for (j in seq_along(neg_vars)) {
      neg_mat[, j] <- neg_mat[, j] * neg_wts[j]
    }
    df$neg <- rowSums(neg_mat, na.rm = TRUE) / sum(neg_wts)
  }
  
  # Scale them
  df$pos_z <- safe_zscore(df$pos)
  df$neg_z <- safe_zscore(df$neg)
  
  return(list(df = df, rwa_out = rwa_out))
}


for (i in 0:15) {
  diff_col <- paste0("cluster_", i, "_diff")
  data[[diff_col]] <- (data[[paste0("cluster_", i, "_", type, "_S2")]] - data[[paste0("cluster_", i, "_", type, "_S1")]]) #/(data[[paste0("cluster_", i, "_", type, "_S2")]] + data[[paste0("cluster_", i, "_", type, "_S1")]])
}
diff_cols <- paste0("cluster_", 0:15, "_diff")



# 1) Convert factor => numeric 0/1
data$listener_choice_num <- as.numeric(as.character(data$listener_choice))

# 2) Suppose your numeric diffs are:
diff_cols <- paste0("cluster_", 0:15, "_diff")

# Build the data frame for RWA
df_rwa <- data.frame(
  listener_choice = data$listener_choice_num,
  data[, diff_cols]
)

# 3) Real Weighted Predictors
res_real <- build_weighted_predictors(
  df = df_rwa,
  outcome = "listener_choice",
  predictors = diff_cols,
  print = TRUE
)
df_real <- res_real$df  # includes pos_z and neg_z
df_real$listener_choice = data$listener_choice

# 4) Fit Real Model with Interaction
model_real <- glm(listener_choice ~ pos_z * neg_z,
                  data = df_real,
                  family = binomial())

summary(model_real)
#plot(allEffects(model_real))

# 5) Collect Real Metrics
pred_real <- predict(model_real, type = "response") > 0.5
acc_real  <- mean(pred_real == (data$listener_choice == 1))
cat("Accuracy for real model:", acc_real, "\n")
coefs_real <- coef(model_real)
# Coeff names: (Intercept), pos_z, neg_z, pos_z:neg_z
real_betas <- coefs_real[c("pos_z", "neg_z", "pos_z:neg_z")]

cat("=== REAL MODEL ===\n")
cat(sprintf("Accuracy = %.3f\n", acc_real))
print(real_betas)

#odds ratio coefs:
cat("Odds ratio for pos_z: ", exp(coefs_real["pos_z"]), "\n")
cat("Odds ratio for neg_z: ", exp(coefs_real["neg_z"]), "\n")
cat("Odds ratio for pos_z:neg_z: ", exp(coefs_real["pos_z:neg_z"]), "\n")

confint_model <- confint.default(model_real)  # or confint(model_real) for profiled CIs
exp(confint_model)

car::vif(model_real)

set.seed(123)
n_boot <- 10000

boot_acc <- numeric(n_boot)
boot_betas <- matrix(NA, nrow = n_boot, ncol = 3)
colnames(boot_betas) <- c("beta_pos", "beta_neg", "beta_int")
pbar <- txtProgressBar(min = 0, max = n_boot, style = 3)
for (b in 1:n_boot) {
  # Shuffle Y
  setTxtProgressBar(pbar, b)
  y_boot <- sample(data$listener_choice_num)
  
  # Build a new DF for RWA
  df_boot <- df_rwa
  df_boot$listener_choice <- y_boot
  
  # Weighted means from RWA
  res_boot <- try(build_weighted_predictors(df_boot, outcome = "listener_choice", predictors = diff_cols),
                  silent = TRUE)
  if (inherits(res_boot, "try-error")) next
  
  df_b <- res_boot$df
  mod_boot <- try(glm(y_boot ~ pos_z * neg_z, data = df_b, family = binomial()), silent = TRUE)
  if (inherits(mod_boot, "try-error")) next
  
  # Store results
  pred_b <- predict(mod_boot, type = "response") > 0.5
  boot_acc[b] <- mean(pred_b == (y_boot == 1))
  
  cfs <- coef(mod_boot)
  # cfs: (Intercept), pos_z, neg_z, pos_z:neg_z
  if (all(c("pos_z", "neg_z", "pos_z:neg_z") %in% names(cfs))) {
    boot_betas[b, "beta_pos"] <- cfs["pos_z"]
    boot_betas[b, "beta_neg"] <- cfs["neg_z"]
    boot_betas[b, "beta_int"] <- cfs["pos_z:neg_z"]
  }
}

boot_acc <- na.omit(boot_acc)
boot_betas <- boot_betas[complete.cases(boot_betas), ]

# P-values
p_val_acc <- mean(boot_acc >= acc_real)

p_val_beta_pos <- mean(abs(boot_betas[, "beta_pos"]) >= abs(real_betas["pos_z"]))
p_val_beta_neg <- mean(abs(boot_betas[, "beta_neg"]) >= abs(real_betas["neg_z"]))
p_val_beta_int <- mean(abs(boot_betas[, "beta_int"]) >= abs(real_betas["pos_z:neg_z"]))

# Plots: 2 x 3
par(mfrow = c(2, 3))

# 1) Accuracy
hist(boot_acc, breaks = 30, main = "Bootstrap Accuracy", xlab = "Accuracy", col = "gray")
abline(v = acc_real, col = "red", lwd = 2)
mtext(sprintf("p = %.4f", p_val_acc), side = 3)

# 2) Beta pos_z
hist(boot_betas[, "beta_pos"], breaks = 30, main = expression(beta[pos] ~ (Bootstrap)), xlab = expression(beta[pos]), col = "skyblue")
abline(v = real_betas["pos_z"], col = "red", lwd = 2)
mtext(sprintf("p = %.4f", p_val_beta_pos), side = 3)

# 3) Beta neg_z
hist(boot_betas[, "beta_neg"], breaks = 30, main = expression(beta[neg] ~ (Bootstrap)), xlab = expression(beta[neg]), col = "salmon")
abline(v = real_betas["neg_z"], col = "red", lwd = 2)
mtext(sprintf("p = %.4f", p_val_beta_neg), side = 3)

# 4) Beta interaction (pos_z:neg_z)
hist(boot_betas[, "beta_int"], breaks = 30, main = expression(beta[pos:neg] ~ (Bootstrap)), xlab = expression(pos_z:neg_z), col = "lightgreen")
abline(v = real_betas["pos_z:neg_z"], col = "red", lwd = 2)
mtext(sprintf("p = %.4f", p_val_beta_int), side = 3)

# 5) Blank or text summary
plot.new()
text(0.5, 0.5, paste("pos_z p =", round(p_val_beta_pos, 4),
                     "\nneg_z p =", round(p_val_beta_neg, 4),
                     "\nint p =",   round(p_val_beta_int, 4)),
     cex = 1.1)

par(mfrow = c(1, 1))



#plot seperately:

png(filename = paste0(graphs_dir, "boot_acc_hist_", type, ".png"), 
    width = 6, height = 6, units = "in", res = 300, bg = "white")

# Create the plot
hist(boot_acc, breaks = 30, main = "", col = "gray", cex.axis = 1.5,
     xlab = "", ylab = "")
abline(v = acc_real, col = "red", lwd = 4)

# Close the graphics device
dev.off()

png(filename = paste0(graphs_dir, "boot_beta_pos_hist_", type, ".png"), 
    width = 6, height = 6, units = "in", res = 300, bg = "white")
# Create the plot
hist(boot_betas[, "beta_pos"], breaks = 30, main = "", col = "skyblue", cex.axis = 1.5,
     xlab = "", ylab = "")
abline(v = real_betas["pos_z"], col = "red", lwd = 4)
# Close the graphics device
dev.off()

png(filename = paste0(graphs_dir, "boot_beta_neg_hist_", type, ".png"), 
    width = 6, height = 6, units = "in", res = 300, bg = "white")
# Create the plot
hist(boot_betas[, "beta_neg"], breaks = 30, main = "", col = "salmon", cex.axis = 1.5,
     xlab = "", ylab = "")
abline(v = real_betas["neg_z"], col = "red", lwd = 4)
# Close the graphics device
dev.off()

png(filename = paste0(graphs_dir, "boot_beta_int_hist_", type, ".png"), 
    width = 6, height = 6, units = "in", res = 300, bg = "white")
# Create the plot
hist(boot_betas[, "beta_int"], breaks = 30, main = "", col = "lightgreen", cex.axis = 1.5,
     xlab = "", ylab = "")
abline(v = real_betas["pos_z:neg_z"], col = "red", lwd = 4)
# Close the graphics device
dev.off()


#___________________________________________________
#_______________________No Clustering_______________
#___________________________________________________

data$avg_corr <- rowMeans(data[grep("minus_cluster.*_corr", names(data))], na.rm = TRUE)
data$avg_mean_comp <- rowMeans(data[grep("minus_cluster.*_mean_comp", names(data))], na.rm = TRUE)
data$avg_corr = safe_zscore(data$avg_corr)
data$avg_mean_comp = safe_zscore(data$avg_mean_comp)

model = glm(listener_choice ~ avg_corr + avg_mean_comp, data = data, family = binomial())
summary(model)

#___________________________________________________
#____________Valence Based Clustering_______________
#___________________________________________________

#clustering for pos: 5, 9, 13 --> this is zero based - so it's actually clusters 6, 10, 14
# clustering for neg: 1, 3, 8 --> this is zero based - so it's actually clusters 2, 4, 9

data$pos_activation_s2 = rowMeans(data[, c("cluster_5_mean_comp_listener_S2", "cluster_9_mean_comp_listener_S2", "cluster_13_mean_comp_listener_S2")], na.rm = TRUE)
data$neg_activation_s2 = rowMeans(data[, c("cluster_1_mean_comp_listener_S2", "cluster_3_mean_comp_listener_S2", "cluster_8_mean_comp_listener_S2")], na.rm = TRUE)
data$pos_activation_s1 = rowMeans(data[, c("cluster_5_mean_comp_listener_S1", "cluster_9_mean_comp_listener_S1", "cluster_13_mean_comp_listener_S1")], na.rm = TRUE)
data$neg_activation_s1 = rowMeans(data[, c("cluster_1_mean_comp_listener_S1", "cluster_3_mean_comp_listener_S1", "cluster_8_mean_comp_listener_S1")], na.rm = TRUE)

#same for corr:
data$pos_mimic_s2_corr = rowMeans(data[, c("cluster_5_corr_S2", "cluster_9_corr_S2", "cluster_13_corr_S2")], na.rm = TRUE)
data$neg_mimic_s2_corr = rowMeans(data[, c("cluster_1_corr_S2", "cluster_3_corr_S2", "cluster_8_corr_S2")], na.rm = TRUE)
data$pos_mimic_s1_corr = rowMeans(data[, c("cluster_5_corr_S1", "cluster_9_corr_S1", "cluster_13_corr_S1")], na.rm = TRUE)
data$neg_mimic_s1_corr = rowMeans(data[, c("cluster_1_corr_S1", "cluster_3_corr_S1", "cluster_8_corr_S1")], na.rm = TRUE)

data$pos_mimic = (data$pos_mimic_s2_corr - data$pos_mimic_s1_corr)
data$neg_mimic = (data$neg_mimic_s2_corr - data$neg_mimic_s1_corr)
data$pos_activation = (data$pos_activation_s2 - data$pos_activation_s1)
data$neg_activation = (data$neg_activation_s2 - data$neg_activation_s1)

#scale:
data$pos_mimic = safe_zscore(data$pos_mimic)
data$neg_mimic = safe_zscore(data$neg_mimic)
data$pos_activation = safe_zscore(data$pos_activation)
data$neg_activation = safe_zscore(data$neg_activation)



#___________________________________________________
#____________activation based model (valenced)______
#___________________________________________________
model = glm(listener_choice ~ pos_activation * neg_activation, data = data, family = binomial())
summary(model)
# Odds Ratios (exp of coefficients)
OR <- exp(coef(model))
print(OR)
# Confidence Intervals for coefficients (on log-odds scale)
confint_model <- confint.default(model)  # or confint(model) for profile CIs
# Confidence Intervals for Odds Ratios
OR_CI <- exp(confint_model)
print(OR_CI)

#___________________________________________________
#____________mimicry based model (valenced)_________
#___________________________________________________

model = glm(listener_choice ~ pos_mimic * neg_mimic, data = data, family = binomial())
summary(model)
OR <- exp(coef(model))
print(OR)
# Confidence Intervals for coefficients (on log-odds scale)
confint_model <- confint.default(model)  # or confint(model) for profile CIs
# Confidence Intervals for Odds Ratios
OR_CI <- exp(confint_model)
print(OR_CI)

#___________________________________________________
#____________plot mimicry based model effects_______
#___________________________________________________

library(ggplot2)
library(patchwork)  # for side-by-side plots
library(gridExtra)
library(lattice)
library(effects)

# Make sure listener_choice is numeric
data$listener_choice <- as.numeric(as.character(data$listener_choice))


# Extract effects and convert to data.frames
eff_pos <- Effect("pos_mimic", model)
df_pos <- as.data.frame(eff_pos)

eff_neg <- Effect("neg_mimic", model)
df_neg <- as.data.frame(eff_neg)

# Define common graphical settings
common_theme <- list(
  par.settings = list(
    axis.text = list(cex = 1.4, font = 2),       # Tick labels bold
    par.main.text = list(cex = 2, font = 2),   # Main title bold
    par.xlab.text = list(cex = 1.9, font = 3),   # X-axis label bold
    par.ylab.text = list(cex = 1.9, font = 3)    # Y-axis label bold
  ),
  scales = list(
    x = list(cex = 1.9, tck = 0.7, font = 3),    # x-axis tick labels bold
    y = list(cex = 1.9, tck = 0.7, font = 3)     # y-axis tick labels bold
  )
)

# POSITIVE MIMICRY PLOT
p1 <- xyplot(fit ~ pos_mimic, data = df_pos,
             type = "l",
             ylim = c(0, 1),
             ylab = "Probability of selecting synopsis 2",
             xlab = "minus_positive_mimicry",
             main = "Positive Mimicry",
             par.settings = common_theme$par.settings,
             scales = common_theme$scales,
             panel = function(x, y, ...) {
               panel.polygon(c(x, rev(x)),
                             c(df_pos$lower, rev(df_pos$upper)),
                             col = rgb(0, 0, 1, 0.2), border = NA)
               panel.lines(x, y, col = "blue", lwd = 2)
               panel.segments(x0 = data$pos_mimic,
                              x1 = data$pos_mimic,
                              y0 = 0,
                              y1 = 0.03,
                              col = "black", lwd = 1.2)
             })

# NEGATIVE MIMICRY PLOT
p2 <- xyplot(fit ~ neg_mimic, data = df_neg,
             type = "l",
             ylim = c(0, 1),
             ylab = "Probability of selecting synopsis 2",
             xlab = "minus_negative_mimicry",
             main = "Negative Mimicry",
             par.settings = common_theme$par.settings,
             scales = common_theme$scales,
             panel = function(x, y, ...) {
               panel.polygon(c(x, rev(x)),
                             c(df_neg$lower, rev(df_neg$upper)),
                             col = rgb(1, 0, 0, 0.2), border = NA)
               panel.lines(x, y, col = "red", lwd = 2)
               panel.segments(x0 = data$neg_mimic,
                              x1 = data$neg_mimic,
                              y0 = 0,
                              y1 = 0.03,
                              col = "black", lwd = 1.2)
             })

# Arrange side by side
grid.arrange(p1, p2, ncol = 2)




#___________________________________________________
#____________summary table__________________________
#___________________________________________________
library(dplyr)

n_clusters <- 16
types <- c("mean_comp_listener", "corr")
summary_list <- list()

for (type in types) {
  for (i in 1:n_clusters) {  # 1-based indexing
    cluster_index <- i - 1  # still need to access the correct column names
    
    S1_col <- paste0("cluster_", cluster_index, "_", type, "_S1")
    S2_col <- paste0("cluster_", cluster_index, "_", type, "_S2")
    
    if (!(S1_col %in% names(data)) | !(S2_col %in% names(data))) {
      message(paste("Skipping missing cluster:", i, "type:", type))
      next
    }
    
    values_s1 <- data[[S1_col]]
    values_s2 <- data[[S2_col]]
    
    row <- data.frame(
      Cluster = i,  # 1-based for display
      Type = type,
      N_S1 = sum(!is.na(values_s1)),
      Mean_S1 = round(mean(values_s1, na.rm = TRUE), 3),
      SD_S1 = round(sd(values_s1, na.rm = TRUE), 3),
      N_S2 = sum(!is.na(values_s2)),
      Mean_S2 = round(mean(values_s2, na.rm = TRUE), 3),
      SD_S2 = round(sd(values_s2, na.rm = TRUE), 3)
    )
    
    summary_list[[length(summary_list) + 1]] <- row
  }
}

summary_stats_table <- bind_rows(summary_list)

# View the table
summary_stats_table
#save as csv:
write.csv(summary_stats_table, "reading_summary_stats.csv", row.names = FALSE)
