import sys
from typing import List
import numpy as np
import pygame
import time
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d

def odwroconeWahadloModelKx(net, isVis):
    width, height = 690, 600

    # Physics constants
    GRAVITY = 900.0
    CART_MASS = 1.0
    ARM1_MASS = 5
    ARM1_INERTIA = 0.1
    MAX_FORCE = 20000
    DESIRED_STATE = [150, 0, 0, 0]  # Desired cart position and angles
    ARM1_L = 75

    # Collision group constants
    COLLISION_GROUP_CART = 1
    COLLISION_GROUP_ARM = 2

    # Initialize visualization and clock
    running = True
    clock = pygame.time.Clock()
    if isVis:
        pygame.init()
        screen = pygame.display.set_mode((width, height))
        draw_options = pymunk.pygame_util.DrawOptions(screen)

    # Physics space
    space = pymunk.Space()
    space.gravity = Vec2d(0.0, GRAVITY)

    # Cart (movable base)
    cart = pymunk.Body(CART_MASS, float('inf'))
    cart.position = (340, 300)
    cart_shape = pymunk.Poly.create_box(cart, size=(50, 10))
    cart_shape.filter = pymunk.ShapeFilter(group=COLLISION_GROUP_CART)  # Set collision group
    space.add(cart, cart_shape)

    # Cart groove joint (to restrict movement to horizontal axis)
    cart_joint = pymunk.GrooveJoint(space.static_body, cart, (670, 300), (10, 300), (0, 0))
    space.add(cart_joint)

    # First arm (attached to cart)
    arm1 = pymunk.Body(ARM1_MASS, ARM1_INERTIA)
    arm1.position = (cart.position.x, cart.position.y - ARM1_L)
    arm1_shape = pymunk.Circle(arm1, 10)
    arm1_shape.filter = pymunk.ShapeFilter(group=COLLISION_GROUP_CART)  # Set collision group
    arm1_joint = pymunk.constraints.PivotJoint(cart, arm1, cart.position)
    space.add(arm1, arm1_shape, arm1_joint)

    # Simulation variables
    fps = 90
    dt = 1.0 / fps
    eT = 0
    prev_cart_x = cart.position.x
    prev_arm1_angle = arm1.angle
    state_error = [0, 0, 0, 0]

    while running:
        # Handle events
        if isVis:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        # Get current state
        cart_x = cart.position.x
        cart_vx = (cart_x - prev_cart_x) / dt
        arm1_angle = arm1.angle
        arm1_vangle = (arm1_angle - prev_arm1_angle) / dt

        # Update previous state
        prev_cart_x = cart_x
        prev_arm1_angle = arm1_angle

        # Neural network input: state error
        current_state = [cart_x, cart_vx, arm1_angle, arm1_vangle]
        error = [(current_state[i] - DESIRED_STATE[i]) for i in range(len(current_state))]
        state_error = [state_error[i] + abs(error[i]) for i in range(len(error))]

        # Apply control law
        control_signal = net.activate(error)
        cart_force = MAX_FORCE * control_signal[0]
        cart_force = max(-MAX_FORCE, min(MAX_FORCE, cart_force))
        if eT < 0.5:
            cart_force = 20
        cart.apply_force_at_world_point((cart_force, 0), cart.position)

        # Update physics
        space.step(dt)

        # Visualization
        if isVis:
            screen.fill(pygame.Color("grey"))  # Set the background color
            def pymunk_to_pygame(pos):
                """Convert Pymunk coordinates to Pygame coordinates."""
                return int(pos[0]), int(height - pos[1])
            
            # Draw the cart
            cart_pos = pymunk_to_pygame((int(cart.position.x), int(height - cart.position.y)))
            pygame.draw.rect(
                screen,
                (255, 165, 0),  # Blue color for the cart
                pygame.Rect(cart_pos[0] - 25, cart_pos[1] - 5, 50, 10)  # Rectangle dimensions
            )

            # Draw the first arm (as an orange circle)
            arm1_pos = pymunk_to_pygame((int(arm1.position.x), int(height - arm1.position.y)))
            pygame.draw.circle(
                screen,
                (255, 165, 0),  # Orange color for the arm
                arm1_pos,
                10  # Radius of the circle
            )

            # Draw a line connecting the cart to the first arm
            pygame.draw.line(
                screen,
                "white",  # Black color
                cart_pos,
                arm1_pos,
                5  # Line width
            )
            pygame.draw.circle(
                screen,
                "purple",  # Orange color for the arm
                cart_pos,
                5  # Radius of the circle
            )
            # Add more custom drawing here if needed (e.g., second arm, ground)

            pygame.display.flip()
            clock.tick(fps)

        # Check simulation end condition
        eT += dt
        if eT > 15:
            running = False

    return state_error
