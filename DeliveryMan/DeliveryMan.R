# Load the library
library("DeliveryMan")

# MAIN FUNCTION
myFunction = function(roads, car, packages){
  # If no package currently held
  if(car$load == 0) {
    car$mem$goal = next_package(roads, car, packages)
  # If package is held
  } else {
    car$mem$goal = packages[car$load, c(3,4)]
  }
  # Determine next move based on the path taken with A*
  move = a_star(roads, car, packages)[1]
  # Stay still if the package and delivery destination are in the same spot
  if (is.null(move)) {
    car$nextMove = 5
    return(car)
  }
  # Otherwise set the direction of the next move based on the retrieved path
  get_moving = as.integer(unlist(strsplit(move, split = ',')))
  if (get_moving[1] > car$x){
    car$nextMove = 6
  } else if (get_moving[1] < car$x) {
    car$nextMove = 4
  } else if (get_moving[2] > car$y) {
    car$nextMove = 8
  } else if (get_moving[2] < car$y) {
    car$nextMove = 2
  }
  return(car)
}

# Function to calculate Manhattan Distance
manhat = function(x, y) sum(abs(x-y)) 

# Function to locate the nearest package
next_package = function(roads, car, packages){
  total_distances = rowSums(abs(sweep(packages[, 1:2], 2, c(car$x, car$y))))
  total_distances[packages[, 5] != 0] = Inf
  return(packages[which.min(total_distances), 1:2])
}

# Decide whether a node should be added to the frontier
to_include = function(candidate, index, frontier, visited){
  if (!index %in% visited){
    if (is.element(index, names(frontier))){
      if (candidate$f < frontier[[index]]$f){
        return(TRUE)
      } else {
      return(FALSE)
      }
    } else {
      return(TRUE)
    }
  }
  return(FALSE)
}

# A* algorithm 
a_star = function(roads, car, packages){
  
  # Set up containers
  frontier = list()
  visited = c()
  
  # Add the origin to the frontier
  origin = list(x = car$x, y = car$y, g = 0, h = manhat(c(car$x, car$y), car$mem$goal), f = manhat(c(car$x, car$y), car$mem$goal), l = c(NULL))
  origin_index = paste(c(origin$x, ',', origin$y), collapse = '')
  frontier[[origin_index]] = origin
  
  # Repeat until the goal is found or frontier is empty
  while (length(frontier) != 0){
    # Select the node with minimal f cost from the frontier and remove it
    expanded = frontier[[1]]
    expanded_index = paste(c(expanded$x, ',', expanded$y), collapse = '')
    frontier[[1]] = NULL
    # Check if expanded node is our goal
    if (expanded$x == car$mem$goal[1] & expanded$y == car$mem$goal[2]){
      return(expanded$l)
    } 
    
    # Examine origin node's neighbours
    # Left neighbour 
    if (expanded$x != 1) {
      index = paste(c(expanded$x-1,',', expanded$y), collapse = '')
      g = roads$hroads[expanded$x-1, expanded$y] + expanded$g
      h = manhat(c(expanded$x-1, expanded$y), car$mem$goal)
      candidate = list(x = expanded$x-1, y = expanded$y, g = g, h = h, f = g + h, l = c(expanded$l, index))
      decision = to_include(candidate, index, frontier, visited)
      if (decision){
        frontier[[index]] = candidate
      }
    }
    # Top neighbour
    if (expanded$y != 10) {
      index = paste(c(expanded$x,',', expanded$y+1), collapse = '')
      g = roads$vroads[expanded$x, expanded$y] + expanded$g
      h = manhat(c(expanded$x, expanded$y+1), car$mem$goal)
      candidate = list(x = expanded$x, y = expanded$y+1, g = g, h = h, f = g + h, l = c(expanded$l, index))
      decision = to_include(candidate, index, frontier, visited)
      if (decision){
        frontier[[index]] = candidate
      }
    }
    # Right neighbour
    if (expanded$x != 10) {
      index = paste(c(expanded$x+1,',', expanded$y), collapse = '')
      g = roads$hroads[expanded$x, expanded$y] + expanded$g
      h = manhat(c(expanded$x+1, expanded$y), car$mem$goal)
      candidate = list(x = expanded$x+1, y = expanded$y, g = g, h = h, f = g + h, l = c(expanded$l, index))
      decision = to_include(candidate, index, frontier, visited)
      if (decision){
        frontier[[index]] = candidate
      }
    }
    # Bottom neighbour
    if (expanded$y != 1) {
      index = paste(c(expanded$x,',', expanded$y - 1), collapse ='')
      g = roads$vroads[expanded$x, expanded$y-1] + expanded$g
      h = manhat(c(expanded$x, expanded$y-1), car$mem$goal)
      candidate = list(x = expanded$x, y = expanded$y-1, g = g, h = h, f = g + h, l = c(expanded$l, index))
      decision = to_include(candidate, index, frontier, visited)
      if (decision){
        frontier[[index]] = candidate
      }
    }
    
    # Add expanded node to the visited set and re-order the frontier
    visited = append(visited, expanded_index)
    frontier = frontier[order(sapply(frontier,'[[',5))]
  }  
}

# TEST FUNCTION
testDM(myFunction)
