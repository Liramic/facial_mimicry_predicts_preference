library(ggplot2)
library(dplyr)
library(lme4)
library(performance)


#cd to current dir:
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# --- LOAD DATA ---
data = read.csv("listening.csv")
data$listener_choice <- as.factor(data$listener_choice)
data = data[data$is_other == 0, ]


safe_zscore <- function(x) {
  (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
}

#these are actually clusters 6, and 10 -- varaible names were zero-based.
data$minus_cluster_5_corr = safe_zscore(data$cluster_5_corr_S2 - data$cluster_5_corr_S1)
data$minus_cluster_9_corr = safe_zscore(data$cluster_9_corr_S2 - data$cluster_9_corr_S1)
data$minus_cluster_5_mean_comp = safe_zscore(data$cluster_5_mean_comp_S2 - data$cluster_5_mean_comp_S1)
data$minus_cluster_9_mean_comp = safe_zscore(data$cluster_9_mean_comp_S2 - data$cluster_9_mean_comp_S1)

# corr model:
filtered_data <- data[complete.cases(data[c("listener_choice", "minus_cluster_5_corr", "minus_cluster_9_corr")]), ]


#check ICC in null model:
null_model <- glmer(listener_choice ~ 1 + (1 | story_1), data = filtered_data, family = "binomial")
summary(null_model)
performance::icc(null_model)

corr_model = glmer(listener_choice ~ minus_cluster_5_corr + minus_cluster_9_corr + (1|story_1), data = filtered_data, family = "binomial")
summary(corr_model)

r2(corr_model)

# Odds ratios for fixed effects
exp(fixef(corr_model))

# Confidence intervals for odds ratios
exp(confint(corr_model, method = "Wald"))


null_model <- glmer(listener_choice ~ 1 + (1 | story_1), data = filtered_data, family = "binomial")
summary(null_model)
anova(null_model, corr_model)  


# activation model:
filtered_data = data[complete.cases(data[c("listener_choice", "minus_cluster_5_mean_comp", "minus_cluster_9_mean_comp")]), ]

activation_model = glmer(listener_choice ~ minus_cluster_5_mean_comp + minus_cluster_9_mean_comp + (1 | story_1), data = filtered_data, family = "binomial")
summary(activation_model)

null_model_activation <- glmer(listener_choice ~ 1 + (1 | story_1), data = filtered_data, family = "binomial")
anova(null_model_activation, activation_model)

r2(activation_model)

library(dplyr)

# Define clusters and types of variables
target_clusters <- c(5, 9)  # 0-based cluster indices for cluster 6 and 10
types <- c("mean_comp", "corr")



# Initialize a list to hold summary rows
summary_list <- list()

# Loop through types and clusters
for (type in types) {
  for (cluster in target_clusters) {
    S1_col <- paste0("cluster_", cluster, "_", type, "_S1")
    S2_col <- paste0("cluster_", cluster, "_", type, "_S2")
    
    if (!(S1_col %in% names(data)) | !(S2_col %in% names(data))) {
      message(paste("Skipping missing columns for cluster", cluster + 1, "type:", type))
      next
    }
    
    values_s1 <- data[[S1_col]]
    values_s2 <- data[[S2_col]]
    
    summary_row <- data.frame(
      Cluster = cluster + 1,  # Convert to 1-based index for display
      Type = type,
      N_S1 = sum(!is.na(values_s1)),
      Mean_S1 = round(mean(values_s1, na.rm = TRUE), 3),
      SD_S1 = round(sd(values_s1, na.rm = TRUE), 3),
      N_S2 = sum(!is.na(values_s2)),
      Mean_S2 = round(mean(values_s2, na.rm = TRUE), 3),
      SD_S2 = round(sd(values_s2, na.rm = TRUE), 3)
    )
    
    summary_list[[length(summary_list) + 1]] <- summary_row
  }
}

# Combine all summaries into a data frame
summary_table <- bind_rows(summary_list)

# View the table
print(summary_table)
#save as csv:
write.csv(summary_table, "listening_summary_stats.csv", row.names = FALSE)
