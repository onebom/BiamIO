# import pygame
# import math
#
# # Initialize Pygame
# pygame.init()
#
# # Set up the screen
# screen_width = 640
# screen_height = 480
# screen = pygame.display.set_mode((screen_width, screen_height))
# pygame.display.set_caption("Follow the Mouse")
#
# # Set up the dot
# dot_radius = 10
# dot_color = (255, 255, 255)
# dot_position = [screen_width/2, screen_height/2]
# dot_speed = 5
#
# # Set up the clock
# clock = pygame.time.Clock()
#
# # Main loop
# running = True
# while running:
#   # Handle events
#   for event in pygame.event.get():
#     if event.type == pygame.QUIT:
#       running = False
#
#   # Move the dot
#   mouse_position = pygame.mouse.get_pos()
#   dx = mouse_position[0] - dot_position[0]
#   dy = mouse_position[1] - dot_position[1]
#   distance = math.sqrt(dx**2 + dy**2)
#   if distance > 0:
#     if distance < 20:
#       # Circle around the mouse pointer
#       dot_velocity = [dot_speed*dy/distance, -dot_speed*dx/distance]
#     else:
#       # Move towards the mouse pointer
#       dot_velocity = [dot_speed*dx/distance, dot_speed*dy/distance]
#     dot_position[0] += dot_velocity[0]
#     dot_position[1] += dot_velocity[1]
#
#   # Draw the dot
#   screen.fill((0, 0, 0))
#   pygame.draw.circle(screen, dot_color, dot_position, dot_radius)
#
#   # Update the screen
#   pygame.display.flip()
#
#   # Tick the clock
#   clock.tick(60)
#
# # Quit Pygame
# pygame.quit()

########################################################################################################################

# import pygame
# import math
#
# # Initialize Pygame
# pygame.init()
#
# # Set up the screen
# screen_width = 640
# screen_height = 480
# screen = pygame.display.set_mode((screen_width, screen_height))
# pygame.display.set_caption("Follow the Mouse")
#
# # Set up the dot
# dot_radius = 10
# dot_color = (255, 255, 255)
# dot_position = [screen_width/2, screen_height/2]
# dot_speed = 5
#
# # Set up the dot's previous position
# prev_dot_position = dot_position.copy()
#
# # Set up the clock
# clock = pygame.time.Clock()
#
# # Main loop
# running = True
# while running:
#   # Handle events
#   for event in pygame.event.get():
#     if event.type == pygame.QUIT:
#       running = False
#
#   # Move the dot
#   mouse_position = pygame.mouse.get_pos()
#   dx = mouse_position[0] - dot_position[0]
#   dy = mouse_position[1] - dot_position[1]
#   distance = math.sqrt(dx**2 + dy**2)
#   if distance > 0:
#     if distance < 50:
#       # Circle around the mouse pointer
#       dot_velocity = [dot_speed*dy/distance, -dot_speed*dx/distance]
#     else:
#       # Move towards the mouse pointer
#       dot_velocity = [dot_speed*dx/distance, dot_speed*dy/distance]
#     dot_position[0] += dot_velocity[0]
#     dot_position[1] += dot_velocity[1]
#
#   # Smooth the dot's movement
#   alpha = 0.2 # weight for the new position
#   dot_position[0] = alpha*dot_position[0] + (1-alpha)*prev_dot_position[0]
#   dot_position[1] = alpha*dot_position[1] + (1-alpha)*prev_dot_position[1]
#   prev_dot_position = dot_position.copy()
#
#   # Draw the dot
#   screen.fill((0, 0, 0))
#   pygame.draw.circle(screen, dot_color, [int(x) for x in dot_position], dot_radius)
#
#   # Update the screen
#   pygame.display.flip()
#
#   # Tick the clock
#   clock.tick(60)
#
# # Quit Pygame
# pygame.quit()

########################################################################################################################
#
# import pygame
# import math
#
# # Initialize Pygame
# pygame.init()
#
# # Set up the screen
# screen_width = 640
# screen_height = 480
# screen = pygame.display.set_mode((screen_width, screen_height))
# pygame.display.set_caption("Follow the Mouse")
#
# # Set up the dot
# dot_radius = 10
# dot_color = (255, 255, 255)
# dot_position = [screen_width/2, screen_height/2]
# dot_speed = 10
#
# # Set up the dot's previous position
# prev_dot_position = dot_position.copy()
#
# # Set up the clock
# clock = pygame.time.Clock()
#
# # Main loop
# running = True
# while running:
#   # Handle events
#   for event in pygame.event.get():
#     if event.type == pygame.QUIT:
#       running = False
#
#   # Move the dot
#   mouse_position = pygame.mouse.get_pos()
#   dx = mouse_position[0] - dot_position[0]
#   dy = mouse_position[1] - dot_position[1]
#   distance = math.sqrt(dx**2 + dy**2)
#   if distance > 0:
#     if distance < 50:
#       # Circle around the mouse pointer
#       dot_velocity = [dot_speed*dy/distance, -dot_speed*dx/distance]
#     else:
#       # Move towards the mouse pointer
#       dot_velocity = [dot_speed*dx/distance, dot_speed*dy/distance]
#     dot_position[0] += dot_velocity[0]
#     dot_position[1] += dot_velocity[1]
#
#   # Smooth the dot's movement
#   alpha = 0.2 # weight for the new position
#   dot_position[0] = alpha*dot_position[0] + (1-alpha)*prev_dot_position[0]
#   dot_position[1] = alpha*dot_position[1] + (1-alpha)*prev_dot_position[1]
#   prev_dot_position = dot_position.copy()
#
#   # Draw the dot
#   screen.fill((0, 0, 0))
#   pygame.draw.circle(screen, dot_color, [int(x) for x in dot_position], dot_radius)
#
#   # Update the screen
#   pygame.display.flip()
#
#   # Tick the clock
#   clock.tick(60)
#
# # Quit Pygame
# pygame.quit()
########################################################################################################################
# import pygame
# import math
#
# # Initialize Pygame
# pygame.init()
#
# # Set up the screen
# screen_width = 640
# screen_height = 480
# screen = pygame.display.set_mode((screen_width, screen_height))
# pygame.display.set_caption("Follow the Mouse")
#
# # Set up the dot
# dot_radius = 10
# dot_color = (255, 255, 255)
# dot_position = [screen_width/2, screen_height/2]
# dot_speed = 10
# dot_velocity = [0, 0]
#
# # Set up the dot's previous position
# prev_dot_position = dot_position.copy()
#
# # Set up the clock
# clock = pygame.time.Clock()
#
# # Main loop
# running = True
# while running:
#   # Handle events
#   for event in pygame.event.get():
#     if event.type == pygame.QUIT:
#       running = False
#
#   # Move the dot
#   mouse_position = pygame.mouse.get_pos()
#   dx = mouse_position[0] - dot_position[0]
#   dy = mouse_position[1] - dot_position[1]
#   distance = math.sqrt(dx**2 + dy**2)
#   if distance > 0:
#     if distance < 50:
#       # Circle around the mouse pointer
#       dot_velocity = [dot_speed*dy/distance, -dot_speed*dx/distance]
#     else:
#       # Move towards the mouse pointer
#       dot_velocity = [dot_speed*dx/distance, dot_speed*dy/distance]
#   dot_position[0] += dot_velocity[0]
#   dot_position[1] += dot_velocity[1]
#
#   # Bounce off the window boundaries
#   if dot_position[0] < dot_radius or dot_position[0] > screen_width - dot_radius:
#     dot_velocity[0] = -dot_velocity[0]
#   if dot_position[1] < dot_radius or dot_position[1] > screen_height - dot_radius:
#     dot_velocity[1] = -dot_velocity[1]
#
#   # Keep the dot within the window boundaries
#   dot_position[0] = max(dot_radius, min(dot_position[0], screen_width - dot_radius))
#   dot_position[1] = max(dot_radius, min(dot_position[1], screen_height - dot_radius))
#
#   # Smooth the dot's movement
#   alpha = 0.2 # weight for the new position
#   dot_position[0] = alpha*dot_position[0] + (1-alpha)*prev_dot_position[0]
#   dot_position[1] = alpha*dot_position[1] + (1-alpha)*prev_dot_position[1]
#   prev_dot_position = dot_position.copy()
#
#   # Draw the dot
#   screen.fill((0, 0, 0))
#   pygame.draw.circle(screen, dot_color, [int(x) for x in dot_position], dot_radius)
#
#   # Update the screen
#   pygame.display.flip()
#
#   # Tick the clock
#   clock.tick(60)
#
# # Quit Pygame
# pygame.quit()
########################################################################################################################
# import pygame
# import math
#
# # Initialize Pygame
# pygame.init()
#
# # Set up the screen
# screen_width = 640
# screen_height = 480
# screen = pygame.display.set_mode((screen_width, screen_height))
# pygame.display.set_caption("Follow the Mouse")
#
# # Set up the dot
# dot_radius = 10
# dot_color = (255, 255, 255)
# dot_position = [screen_width/2, screen_height/2]
# dot_speed = 30
# dot_velocity = [0, 0]
#
# # Set up the dot's previous position
# prev_dot_position = dot_position.copy()
#
# # Set up the clock
# clock = pygame.time.Clock()
#
# # Main loop
# running = True
# while running:
#   # Handle events
#   for event in pygame.event.get():
#     if event.type == pygame.QUIT:
#       running = False
#
#   # Move the dot
#   mouse_position = pygame.mouse.get_pos()
#   dx = mouse_position[0] - dot_position[0]
#   dy = mouse_position[1] - dot_position[1]
#   distance = math.sqrt(dx**2 + dy**2)
#   if distance > 0:
#     if distance < 50:
#       # Circle around the mouse pointer
#       dot_velocity = [dot_speed*dy/distance, -dot_speed*dx/distance]
#     else:
#       # Move towards the mouse pointer
#       dot_velocity = [dot_speed*dx/distance, dot_speed*dy/distance]
#   else:
#     # Maintain previous velocity if mouse is not in the window
#     dot_velocity = [0, 0]
#   dot_position[0] += dot_velocity[0]
#   dot_position[1] += dot_velocity[1]
#
#   # Bounce off the window boundaries
#   if dot_position[0] < dot_radius or dot_position[0] > screen_width - dot_radius:
#     dot_velocity[0] = -dot_velocity[0]
#   if dot_position[1] < dot_radius or dot_position[1] > screen_height - dot_radius:
#     dot_velocity[1] = -dot_velocity[1]
#
#   # Keep the dot within the window boundaries
#   dot_position[0] = max(dot_radius, min(dot_position[0], screen_width - dot_radius))
#   dot_position[1] = max(dot_radius, min(dot_position[1], screen_height - dot_radius))
#
#   # Smooth the dot's movement
#   alpha = 0.2 # weight for the new position
#   dot_position[0] = alpha*dot_position[0] + (1-alpha)*prev_dot_position[0]
#   dot_position[1] = alpha*dot_position[1] + (1-alpha)*prev_dot_position[1]
#   prev_dot_position = dot_position.copy()
#
#   # Draw the dot
#   screen.fill((0, 0, 0))
#   pygame.draw.circle(screen, dot_color, [int(x) for x in dot_position], dot_radius)
#
#   # Update the screen
#   pygame.display.flip()
#
#   # Tick the clock
#   clock.tick(60)
#
# # Quit Pygame
# pygame.quit()
########################################################################################################################
import pygame
import math

# Initialize Pygame
pygame.init()

# Set up the screen
screen_width = 1280
screen_height = 720
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Follow the Mouse")

# Set up the dot
dot_radius = 10
dot_color = (255, 255, 255)
dot_position = [screen_width/2, screen_height/2]
dot_speed = 30
dot_velocity = [0, 0]

# Set up the dot's previous position
prev_dot_position = dot_position.copy()

# Set up the clock
clock = pygame.time.Clock()

# Main loop
running = True
while running:
  # Handle events
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      running = False

  # Move the dot
  mouse_position = pygame.mouse.get_pos()
  dx = mouse_position[0] - dot_position[0]
  dy = mouse_position[1] - dot_position[1]
  distance = math.sqrt(dx**2 + dy**2)

  # Gradually increase dot speed based on distance to mouse pointer
  min_speed = 20
  max_speed = 100
  dot_speed = min(max_speed, max(distance / 10, min_speed))

  if distance > 0:
    if distance < 100:
      # Circle around the mouse pointer
      dot_velocity = [dot_speed*dy/distance, -dot_speed*dx/distance]
    else:
      # Move towards the mouse pointer
      dot_velocity = [dot_speed*dx/distance, dot_speed*dy/distance]
  else:
    # Maintain previous velocity if mouse is not in the window
    dot_velocity = [0, 0]
  dot_position[0] += dot_velocity[0]
  dot_position[1] += dot_velocity[1]

  # Bounce off the window boundaries
  if dot_position[0] < dot_radius or dot_position[0] > screen_width - dot_radius:
    dot_velocity[0] = -dot_velocity[0]
  if dot_position[1] < dot_radius or dot_position[1] > screen_height - dot_radius:
    dot_velocity[1] = -dot_velocity[1]

  # Keep the dot within the window boundaries
  dot_position[0] = max(dot_radius, min(dot_position[0], screen_width - dot_radius))
  dot_position[1] = max(dot_radius, min(dot_position[1], screen_height - dot_radius))

  # Smooth the dot's movement
  alpha = 0.2 # weight for the new position
  dot_position[0] = alpha*dot_position[0] + (1-alpha)*prev_dot_position[0]
  dot_position[1] = alpha*dot_position[1] + (1-alpha)*prev_dot_position[1]
  prev_dot_position = dot_position.copy()

  # Draw the dot
  screen.fill((0, 0, 0))
  pygame.draw.circle(screen, dot_color, [int(x) for x in dot_position], dot_radius)

  # Update the screen
  pygame.display.flip()

  # Tick the clock
  clock.tick(60)

# Quit Pygame
pygame.quit()
