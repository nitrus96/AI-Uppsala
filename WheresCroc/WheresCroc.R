library(WheresCroc)

# MAIN FUNCTION
myFunction = function(moveInfo, readings, positions, edges, probs) {
  
  #print('START TURN')
  
  # Assume that Croc has a uniform probability of being in any of the holes when the game starts, excluding the ranger and backpacker positions
  if (is.null(moveInfo$mem$current_state) | moveInfo$mem$status == 1) {
    init_state = rep(1, times = 40)
    init_state[positions[1:3]] = 0
    init_state = (init_state / 37) %*% emissions(readings, probs)
    moveInfo$mem$current_state = init_state / sum(init_state)

  # Get transition matrix and neighbour list
    moveInfo$mem$t_matrix = transition_matrix(edges)
    moveInfo$mem$neighbours = apply(moveInfo$mem$t_matrix, MARGIN = 1, function(x) which(x != 0))
  
  # Subsequent game moves  
  } else {
  
  # Get emission matrix and calculate forward probabilities
  e_matrix = emissions(readings, probs)
  forward_state = moveInfo$mem$current_state %*% moveInfo$mem$t_matrix %*% e_matrix

  
  # Inspect backpacker positions
  # If a backpacker is eaten, we know with certainty the croc is in that hole, and vice-versa
  # First backpacker
  if (!is.na(positions[1])){
    if (positions[1] < 0) {
      #cat('Backpacker 2 is dead in hole', abs(positions[2]), fill = TRUE)
      forward_state[abs(positions[1])] = 1    
    } else {
      forward_state[positions[1]] = 0
    }
  }
  # Second backpacker
  if (!is.na(positions[2])){
    if (positions[2] < 0) {
      #cat('Backpacker 2 is dead in hole', abs(positions[2]), fill = TRUE)
      forward_state[abs(positions[2])] = 1    
    } else {
      forward_state[positions[2]] = 0
    }
  } 
  
  # Normalise forward probabilities
  moveInfo$mem$current_state = forward_state/sum(forward_state)
  
  }  
  
  # Retrieve the most probabable Croc location
  croc_position = which.max(moveInfo$mem$current_state)

  # Move towards the most probabable Croc location
  next_move = path(moveInfo, positions, croc_position)
  #cat('Predicted Croc location: ', croc_position, fill = TRUE)
  #cat('Ranger location: ', positions[3], fill = TRUE)
  #cat('Path to Croc: ', next_move, length(next_move), fill = TRUE)

  if (length(next_move) == 0){
    moveInfo$moves = c(0, 0)
    moveInfo$mem$current_state[positions[3]] = 0
  } else if (length(next_move) == 1) {
    moveInfo$moves = c(next_move[1],0)
    moveInfo$mem$current_state[next_move[1]] = 0
  } else {
    moveInfo$moves = c(next_move[1], next_move[2])
  }
  
  # Return updated moveInfo
  moveInfo$mem$status = moveInfo$mem$status + 1
  
  return(moveInfo)
}

# Return a 40x40 transition matrix from the list of edges
transition_matrix = function(edges){
  
  t_matrix = matrix(0, nrow = 40, ncol = 40)
  # For each node, the probability of transitioning to one of its neighbours is 1/(total number of neighbours + 1)
  for (n in 1:40) {
    neighbours = c(edges[which(edges[,1] == n),2], edges[which(edges[,2] == n),1], n)
    t_matrix[n, neighbours] = 1/length(neighbours) 
  }
  return(t_matrix)
}

# Return a matrix containing probabilities of observing specific readings given that Croc is in one of the holes
emissions = function(readings, probs) {
  
  salinity = apply(probs$salinity, MARGIN = 1, function(x) dnorm(readings[1], x[1], x[2]))
  phosphate = apply(probs$phosphate, MARGIN = 1, function(x) dnorm(readings[2], x[1], x[2]))
  nitrogen = apply(probs$nitrogen, MARGIN = 1, function(x) dnorm(readings[3], x[1], x[2]))
  # Each reading is independent so they can be multiplied
  total_prob = salinity * phosphate * nitrogen
  emission_matrix = diag(x = total_prob)
  return(emission_matrix)
}

# Breadth first searching algorithm to determine the shortest path
path = function(moveInfo, positions, croc_position){
  
  start_node = positions[3]
  visited = c(positions[3])
  frontier = list()
  frontier[[as.character(start_node)]] = list(node = start_node, path = c(NULL))
  
  while (TRUE) {
    node = frontier[[1]]
    frontier[[as.character(node$node)]] = NULL
    if (node$node == croc_position) {
      return(c(node$path[node$path != start_node], croc_position))
    }
    for (edge in moveInfo$mem$neighbours[[node$node]]){
      if (!is.element(edge, visited)) {
        visited = c(visited, edge)
        frontier[[as.character(edge)]] = list(node = edge, path = c(node$path, node$node))
      }
    }
  } 
}

testWC(myFunction)