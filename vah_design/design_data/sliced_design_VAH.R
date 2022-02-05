library(SLHD)
library(lhs)

n <- 500 # sample size
m <- 50 # sample size in each slice
theta_no <- 15 # number of parameters

sliced_design_VAH <- maximinSLHD(t = n/m, m = m, k = theta_no)$StandDesign

# Write the original standardized design 
write.table(sliced_design_VAH, 'sliced_VAH_test.txt')

lb <- c(10, -0.7, 0.5, 0, 0.3, 0.135, 0.13, 0.01, -2, -1, 0.01, 0.12, 0.025, -0.8, 0.3)
ub <- c(30, 0.7, 1.5, 1.7, 2, 0.165, 0.3, 0.2, 1, 2, 0.25, 0.3, 0.15, 0.8, 1)
design <- sliced_design_VAH[, -1]
design_01 <- sliced_design_VAH[, -1]
for(i in 1:length(lb)){
  design[, i] <- (ub[i] - lb[i])*design_01[,i] + lb[i]
}

design <- cbind(design, rep(0.05, 500))
colnames <- c('Pb_Pb', 'Mean', 'Width', 'Dist', 'Flactutation', 'Temp', 'Kink', 'eta_s', 'Slope_low', 'Slope_high',
  'Max', 'Temp_peak', 'Width_peak', 'Asym_peak', 'R', 'tau_initial')
colnames(design) <- colnames

write.table(design, row.names = F, 'sliced_VAH_090321_test.txt')

# Observe first slice
pairs(design[1:50,],  pch = 19)
for (col in 1:ncol(design)) {
  hist(design[,col])
}