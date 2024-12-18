# libraries
from typing import List
import pygame                   # Used for visualization and event handling
import pymunk                   # Physics engine for simulating rigid body dynamics
import pymunk.pygame_util       # Helper for drawing Pymunk objects using Pygame
import pymunk.constraints       # Used for creating physical constraints
from pymunk.vec2d import Vec2d  # Vector operations for 2D physics

def odwroconeWahadloModelKx(net, isVis: bool):
    """
    Simulates an inverted pendulum system controlled by a neural network.

    Args:
        net: The neural network used to control the system. It takes the error vector as input
             and outputs the control signal to stabilize the pendulum.
        isVis (bool): Whether to visualize the simulation (True = visualization enabled).

    Returns:
        List[float]: The cumulative error metrics of the system, where each value corresponds
                     to the sum of absolute errors for different state variables.
    """

    # Constants
    WIDTH, HEIGHT = 690, 600        # Dimensions of the simulation window (in pixels)
    FPS = 90                        # Frames per second for the simulation
    DT = 1.0 / FPS                  # Time step for the physics engine (in seconds)
    MAX_FORCE = 20000               # Maximum allowable force that can be applied to the cart (in arbitrary units)
    INIT_FORCE = 100                # Initial perturbation force applied to the cart at the start of the simulation (in arbitrary units)
    GRAVITY = 900.0                 # Gravitational force in arbitrary units

    # Target state
    DESIRED_STATES = [-150, 0, 0, 0, 0, 0]
    """
    DESIRED_STATES represents the ideal state of the system:
        [ cart_x -- cart_vx -- arm1_angle -- arm1_angular_velocity -- arm2_angle -- arm2_angular_velocity ]
    cart_x                  --> Target horizontal position of the cart on the x-axis.
    cart_vx                 --> Target velocity of the cart along the x-axis (should ideally be 0).
    arm1_angle              --> Desired rotational angle of the first pendulum arm (upright position = 0).
    arm1_angular_velocity   --> Desired angular velocity of the first pendulum arm (should ideally be 0).
    arm2_angle              --> Desired rotational angle of the second pendulum arm (upright position = 0).
    arm2_angular_velocity   --> Desired angular velocity of the second pendulum arm (should ideally be 0).
    """

    # Pygame and Pymunk initialization
    if isVis:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))       # Set up the simulation window
        draw_options = pymunk.pygame_util.DrawOptions(screen)   # Helper for drawing Pymunk objects
        font = pygame.font.SysFont("Arial", 16)                 # Font for optional on-screen text

    # Physics Space setup
    space = pymunk.Space()                  # Create a Pymunk physics space
    space.gravity = Vec2d(0.0, GRAVITY)     # Set gravity to act downward

    # Cart setup (platform on which the pendulum arms are mounted)
    cart_body = pymunk.Body(10, float("inf"))                                   # Body with high mass (10 units) and infinite moment of inertia
    cart_body.position = 340, 400                                               # Initial position of the cart
    cart_shape = pymunk.Poly.create_box(cart_body, size=(50, 10), radius=1)     # Create a rectangular cart shape
    cart_shape.filter = pymunk.ShapeFilter(group=1)                             # Assign a collision group to the cart

    move_joint = pymunk.GrooveJoint(
        space.static_body, cart_body, (670, 400), (10, 400), (0, 0)
    )  # Groove joint keeps the cart constrained to the horizontal axis

    # First pendulum arm setup
    arm1_body = pymunk.Body(1, pymunk.moment_for_box(1, (10, 100)))             # Body with mass 1 unit and calculated moment of inertia
    arm1_body.position = 340, 350                                               # Initial position of the first pendulum arm
    arm1_shape = pymunk.Poly.create_box(arm1_body, size=(10, 100), radius=1)    # Create a rectangular shape for the arm
    arm1_shape.filter = pymunk.ShapeFilter(group=1)                             # Assign a collision group to the arm
    arm1_joint = pymunk.constraints.PivotJoint(cart_body, arm1_body, (340, 400))# Pivot joint connects the arm to the cart

    # Second pendulum arm setup
    arm2_body = pymunk.Body(1, pymunk.moment_for_box(1, (10, 50)))              # Body with mass 1 unit and calculated moment of inertia
    arm2_body.position = 340, 275                                               # Initial position of the second pendulum arm
    arm2_shape = pymunk.Poly.create_box(arm2_body, size=(10, 50), radius=1)     # Create a rectangular shape for the arm
    arm2_shape.filter = pymunk.ShapeFilter(group=1)                             # Assign a collision group to the arm
    arm2_joint = pymunk.constraints.PivotJoint(arm1_body, arm2_body, (340, 300))# Pivot joint connects the second arm to the first

    # Add all physical elements to the simulation space
    space.add(cart_body, cart_shape, move_joint, arm1_body, arm1_shape, arm1_joint, arm2_body, arm2_shape, arm2_joint)

    # Simulation variables
    running = True                  # Control flag for the simulation loop
    clock = pygame.time.Clock()     # Clock to control simulation speed
    elapsed_time = 0                # Tracks the total simulation time
    previous_state = {              # Stores the state of the system in the previous frame
        "cart_x": 340,
        "arm1_angle": 0,
        "arm2_angle": 0
    }
    cumulative_error = [0, 0, 0, 0, 0, 0]  # Tracks cumulative errors for each state variable

    # Main simulation loop
    while running:
        # Handle events (e.g., closing the window)
        if isVis:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        # Extract current state of the system
        cart_x = cart_shape.body.position[0]                                        # Horizontal position of the cart
        cart_vx = (cart_x - previous_state["cart_x"]) / DT                          # Velocity of the cart
        arm1_angle = arm1_shape.body.angle                                          # Angle of the first pendulum arm
        arm2_angle = arm2_shape.body.angle                                          # Angle of the second pendulum arm
        arm1_angular_velocity = (arm1_angle - previous_state["arm1_angle"]) / DT    # Angular velocity of the first arm
        arm2_angular_velocity = (arm2_angle - previous_state["arm2_angle"]) / DT    # Angular velocity of the second arm

        # Update previous state
        previous_state.update({
            "cart_x": cart_x,
            "arm1_angle": arm1_angle,
            "arm2_angle": arm2_angle
        })

        # Construct the state vector
        state = (cart_x, cart_vx, arm1_angle, arm1_angular_velocity, arm2_angle, arm2_angular_velocity)

        # Compute error between the current and desired states
        error = [(state[i] - DESIRED_STATES[i]) / (100 if i < 2 else 1) for i in range(6)]
        cumulative_error = [cumulative_error[i] + abs(error[i]) for i in range(6)]

        # Neural network control: Calculate control signal based on error
        control_signal = net.activate(error)
        force = 2000 * control_signal[0]                # Scale the control signal to produce a force
        force = max(min(force, MAX_FORCE), -MAX_FORCE)  # Limit the force within the allowable range

        # Apply an initial perturbation force at the start of the simulation
        if elapsed_time < 0.03:
            force = INIT_FORCE

        # Apply the computed force to the cart
        cart_body.apply_force_at_world_point((force, 0), cart_body.position)

        # Update the physics simulation
        space.step(DT)

        # Visualization (if enabled)
        if isVis:
            screen.fill(pygame.Color("white"))  # Clear the screen
            space.debug_draw(draw_options)      # Draw all elements in the space
            pygame.display.flip()               # Update the display
            clock.tick(FPS)                     # Maintain the desired FPS

        # Increment elapsed simulation time
        elapsed_time += DT

        # Stop the simulation after 15 seconds
        if elapsed_time > 15:
            running = False

    return cumulative_error
