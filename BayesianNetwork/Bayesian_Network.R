library(Diagnostics) 

# Create a Bayesian network from the given historical data
learn = function(hist_data = hist){
  bayes_network = list()

  # Index [, 1] is P = 0, index [, 2] is P = 1
  
  # Pneumonia statistical model
  bayes_network$Pn = matrix(ncol = 2)
  bayes_network$Pn[, 1:2] = c(length(which(hist_data$Pn == 0)) / 10000,
                              length(which(hist_data$Pn == 1)) / 10000)

  # Temperature statistical model
  # Conditonal normal distribution if Pn == 0
  zero_counts = hist_data[which(hist_data$Pn == 0), 'Te']
  zero_dist = function(x) dnorm(x, mean(zero_counts), sd(zero_counts))
  # If Pn == 1
  one_counts = hist_data[which(hist_data$Pn == 1), 'Te']
  one_dist = function(x) dnorm(x, mean(one_counts), sd(one_counts))
  bayes_network$Te = list(zero_dist, one_dist)
  
  # VTB statistical model
  bayes_network$VTB = matrix(ncol = 2)
  bayes_network$VTB[, 1:2] = c(length(which(hist_data$VTB == 0)) / 10000,
                              length(which(hist_data$VTB == 1)) / 10000)

  # Tuberculosis statistical model
  bayes_network$TB = matrix(nrow = 2, ncol = 3)
  bayes_network$TB[, 3] = 0:1
  colnames(bayes_network$TB)[3] = 'VTB'
  for (i in 1:2) {
    for (j in 1:2) {
      subset = hist_data[which(hist_data$VTB == i-1),]
      bayes_network$TB[i, j] = length(which(subset$TB == j-1)) / length(subset$TB)
    }
  }

  # Smokes statistical model
  bayes_network$Sm = matrix(ncol = 2)
  bayes_network$Sm[, 1:2] = c(length(which(hist_data$Sm == 0)) / 10000,
                              length(which(hist_data$Sm == 1)) / 10000)
  
  # Lung cancer statistical model
  bayes_network$LC = matrix(nrow = 2, ncol = 3)
  bayes_network$LC[, 3] = 0:1
  colnames(bayes_network$LC)[3] = 'Sm'
  for (i in 1:2) {
    for (j in 1:2) {
      subset = hist_data[which(hist_data$Sm == i-1),]
      bayes_network$LC[i, j] = length(which(subset$LC == j-1)) / length(subset$LC)
    }
  }
  
  # Bronchitis statistical model
  bayes_network$Br = matrix(nrow = 2, ncol = 3)
  bayes_network$Br[, 3] = 0:1 
  colnames(bayes_network$Br)[3] = 'Sm'
  for (i in 1:2) {
    for (j in 1:2) {
      subset = hist_data[which(hist_data$Sm == i-1),]
      bayes_network$Br[i, j] = length(which(subset$Br == j-1)) / length(subset$Br)
    }
  }

  # X-Ray result statistical model
  bayes_network$XR = matrix(nrow = 8, ncol = 5)
  bayes_network$XR[, 3:5] = c(c(rep(0, 4), rep(1, 4)), rep(c(0,0,1,1), 2), rep(0:1, 4))
  colnames(bayes_network$XR)[3:5] = c('Pn', 'TB', 'LC')
  for (i in 1:8) {
    for (j in 1:2) {
      subset = hist_data[which(hist_data$Pn == bayes_network$XR[i,'Pn'] &
                                 hist_data$TB == bayes_network$XR[i,'TB'] &
                                 hist_data$LC == bayes_network$XR[i,'LC']),]
      bayes_network$XR[i, j] = length(which(subset$XR == j-1)) / length(subset$XR)
    }
  }

  # Dyspnea statistical model
  bayes_network$Dy = matrix(nrow = 4, ncol = 4)
  bayes_network$Dy[, 3:4] = c(c(rep(0, 2), rep(1, 2)), rep(0:1, 2))
  colnames(bayes_network$Dy)[3:4] = c('LC', 'Br')
  for (i in 1:4) {
    for (j in 1:2) {
      subset = hist_data[which(hist_data$LC == bayes_network$Dy[i,'LC'] & 
                                 hist_data$Br == bayes_network$Dy[i,'Br']),]
      bayes_network$Dy[i, j] = length(which(subset$Dy == j-1)) / length(subset$Dy)
    }
  }
  return(bayes_network)
}

# Calculate disease probabilities from the Bayesian network
diagnose = function(network, cases_data = cases){
  diagnoses = matrix(nrow = 10, ncol = 4)

  # Choose case to diagnose
  for (c in 1:10){
    current_case = cases_data[c,]
    # Determine which values are missing
    unknown = which(is.na(current_case))
    # Define a matrix of 1000 samples
    samples = matrix(nrow = 1000, ncol = 9)
    colnames(samples) = colnames(current_case)
    samples = rbind(current_case, samples)
    # Randomly initialise the starting sample
    samples[1, unknown] = sample(c(1,0), 4, replace = TRUE)
    samples[, -unknown] = current_case[, -unknown]

    # Metropolis in Gibbs sampling for 1000 iterations
    for (no in 1:1000){
      new_vals = samples[no,]
      p_old = get_probability(new_vals, network)
      
      # Use f(unk) = 0 if unk = 1 and vice versa as a candidate function
      for (unk in unknown){
        if (new_vals[unk] == 0){
          new_vals[unk] = 1
        } else if (new_vals[unk] == 1) {
          new_vals[unk] = 0
        }
        # Get sample probabiltiies
        p_new = get_probability(new_vals, network)
        # Determine whether the candidate value should be updated
        if (p_new < p_old){
          prelim_prob = p_new / p_old
          if (prelim_prob  < runif(1)) {
            new_vals[unk] = abs(new_vals[unk] - 1)
          } else {
            p_old = p_new
          }
        } else {
          p_old = p_new
        }
      }
      # Update the sample matrix
      samples[no+1, ] = new_vals
    }
    # Discard first 100 samples
    samples = samples[-100, ]
    # Pneumonia
    diagnoses[c, 1] = length(which(samples$Pn == 1)) / length(samples$Pn)
    # TB 
    subset_tb = samples[which(samples$VTB == current_case$VTB),]
    diagnoses[c, 2] = length(which(subset_tb$TB == 1)) / length(subset_tb$TB)
    # LC
    subset_lc = samples[which(samples$Sm == current_case$Sm),]
    diagnoses[c, 3] = length(which(subset_lc$LC == 1)) / length(subset_lc$LC)
    # Br
    subset_br = samples[which(samples$Sm == current_case$Sm),]
    diagnoses[c, 4] = length(which(subset_br$Br == 1)) / length(subset_lc$Br)
  }
  return(diagnoses)
}

# Calculate samples probabilities
get_probability = function(case, network) {
  p_Pn = network$Pn[,case$Pn + 1]
  p_Te = network$Te[[case$Pn + 1]](case$Te)
  p_VTB = network$VTB[, case$VTB + 1]
  p_TB = network$TB[which(network$TB[,'VTB'] == case$VTB), case$TB +1]
  p_Sm = network$Sm[, case$Sm + 1]
  P_LC = network$LC[which(network$LC[,'Sm'] == case$Sm), case$LC +1]

  p_Br = network$Br[which(network$LC[,'Sm'] == case$Sm), case$Br +1]
  p_XR = network$XR[which(network$XR[,'Pn'] == case$Pn & 
                            network$XR[,'TB'] == case$TB &
                            network$XR[,'LC'] == case$LC), case$XR +1]
  p_Dy = network$Dy[which(network$Dy[,'LC'] == case$LC &
                            network$Dy[,'Br'] == case$Br), case$Dy +1]
  
  prob = p_Pn * p_Te * p_VTB * p_TB * p_Sm * P_LC * p_Br * p_XR * p_Dy
  return(prob)
}

runDiagnostics(learn, diagnose, verbose = 2)
