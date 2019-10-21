library(WobbiesWorld)

myFunction = function(maze = makeWobbiesWorld()){
  
  # Chose which action to perform under the current policy
  decideAction = function(cont, maze) {
    
    # Set up a Q table and epsilon value
    if (is.null(cont$Q)){
      wobbie_pos = matrix(rep(rep(1:27), 27))
      monster_pos = matrix(rep(1:27, each = 27))
      moves = matrix(0L,nrow= 27 *27, ncol = 4)
      cont$Q = cbind(wobbie_pos, monster_pos, moves)
      colnames(cont$Q) = c('W','M','2','4','6','8')
      # Set up variables
      cont$epsilon = 1
      cont$lambda = 0.06
      cont$gamma = 0.5
      cont$doRand = T
    }
    # Store previous maze state in memory
    cont$p_maze = maze
    
    # Retrieve indices of Wobbie and monster
    cont$old_wobbie_i = which(maze$maze$x == maze$wobbie[1] & maze$maze$y == maze$wobbie[2])
    cont$old_monster_i = which(maze$maze$x == maze$monster1[1] & maze$maze$y == maze$monster1[2])
    
    # Make a random move according to the e-greedy policy
    if (runif(1) > cont$epsilon & cont$doRand == T){
      next_move =  sample(c(2, 4, 6, 8), 1)
      return(list(move = next_move, control = cont))
    }
    # Or else choose an action with the highest Q value
    else{
      # Subset states
      sub_state = cont$Q[which(cont$Q[,'W'] == cont$old_wobbie_i & cont$Q[,'M'] == cont$old_monster_i), ]
      # Select next action 
      next_action = names(which.max(sub_state[3:7]))
      return(list(move = strtoi(next_action), control = cont))
    }
  }
  
  # Update the Q table
  update = function(cont, maze) {
    
    # Choose a' from new state using the e-greedy policy
    new_wobbie_i = which(maze$maze$x == maze$wobbie[1] & maze$maze$y == maze$wobbie[2])
    new_monster_i = which(maze$maze$x == maze$monster1[1] & maze$maze$y == maze$monster1[2])
    if (runif(1) > cont$epsilon & cont$doRand == T){
      next_a = toString(sample(c(2, 4, 6, 8), 1))
    } else {
      sub_state = cont$Q[which(cont$Q[,'W'] == new_wobbie_i & cont$Q[,'M'] == new_monster_i), ]
      next_a = names(which.max(sub_state[3:7]))
    }
    # Store Q(s, a)
    previous_q_val = cont$Q[which(cont$Q[,'W'] == cont$old_wobbie_i &
                                           cont$Q[,'M'] == cont$old_monster_i), toString(maze$lastAction)]
    # Store Q(s', a')
    next_q_val = cont$Q[which(cont$Q[, 'W'] == new_wobbie_i &
                                cont$Q[, 'M'] == new_monster_i), next_a]
    # Calculate update
    new_q_val = previous_q_val + (cont$lambda * (maze$reward + (cont$gamma * next_q_val) - previous_q_val))
    cont$Q[which(cont$Q[, 'W'] == cont$old_wobbie_i &
                   cont$Q[, 'M'] == cont$old_monster_i), toString(maze$lastAction)] = new_q_val
    # Epsilon decay
    if (cont$epsilon >= 0.1){
      cont$eplsion = cont$epsilon - 0.01
    }
    return(cont)
  }
  list(decideAction = decideAction, update = update, doRand = T)
}


runWW(myFunction)