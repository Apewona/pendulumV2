import sys
from typing import List
import numpy as np
import pygame
import time

import pymunk
import pymunk.pygame_util
import pymunk.constraints

from pymunk.vec2d import Vec2d
    
def odwroconeWahadloModelKx(net, isVis):
    width, height = 690, 600

    COLLTYPE_DEFAULT = 0
    COLLTYPE_MOUSE = 1
    COLLTYPE_BALL = 2

    running = True
    clock = pygame.time.Clock()

    if isVis:
        pygame.init()
        screen = pygame.display.set_mode((width, height))
        draw_options = pymunk.pygame_util.DrawOptions(screen)
        font = pygame.font.SysFont("Arial", 16)

    space = pymunk.Space()
    space.gravity = Vec2d(0.0, 900.0)

    # Główna platforma
    body = pymunk.Body(10, float('inf'))  # Duża masa dla stabilności platformy
    body.position = 340, 400
    shape = pymunk.Poly.create_box(
        body, size=(50, 10), radius=1
    )
    shape.collision_type = COLLTYPE_DEFAULT
    shape.filter = pymunk.ShapeFilter(group=1)

    move_joint = pymunk.GrooveJoint(
        space.static_body, body, (670, 400), (10, 400), (0, 0)
    )

    # Pierwsze ramię wahadła
    wahadlo = pymunk.Body(1, pymunk.moment_for_box(1, (10, 100)))  
    wahadlo.position = 340, 350
    wahadloShape = pymunk.Poly.create_box(
        wahadlo, size=(10, 100), radius=1
    )
    wahadloShape.collision_type = COLLTYPE_DEFAULT
    wahadloShape.filter = pymunk.ShapeFilter(group=1)

    c = pymunk.constraints.PivotJoint(body, wahadlo, (340, 400))


    # Drugie ramię wahadła
    wahadlo2 = pymunk.Body(1, pymunk.moment_for_box(1, (10, 50)))  
    wahadlo2.position = 340, 275
    wahadlo2Shape = pymunk.Poly.create_box(
        wahadlo2, size=(10, 50), radius=1
    )
    wahadlo2Shape.collision_type = COLLTYPE_DEFAULT
    wahadlo2Shape.filter = pymunk.ShapeFilter(group=1)

    c2 = pymunk.constraints.PivotJoint(wahadlo, wahadlo2, (340, 300))
    # c2.max_bias = 0
    # c2.max_force = 5000000

    # # Dodanie tłumienia w ruchach wahadła
    # damp1 = pymunk.constraints.DampedRotarySpring(body, wahadlo, 0, 200, 10)
    # damp2 = pymunk.constraints.DampedRotarySpring(wahadlo, wahadlo2, 0, 200, 10)


    
    # Dodanie elementów do przestrzeni
    space.add(body, shape, wahadlo, wahadloShape, move_joint, c, wahadlo2, wahadlo2Shape, c2)

    start_time = 0
    eT=0
    cartX_1=340; #wartosc pozycji x z poprzedniego kroku
    cartA_1=0; #wartosc kata z poprzedniego kroku
    arm2_A_1 = 0; 

    fps = 90
    dt = 1.0 / fps
    sE = [0, 0, 0, 0, 0, 0] #suma bledow
    
    while running:

        if isVis==True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

##        keys = pygame.key.get_pressed()
##        left = int(keys[pygame.K_LEFT])
##        right = int(keys[pygame.K_RIGHT])

        
        #odczytujemy wektor stanu z silnika fizyki
        cartX = shape.body.position[0]
        cartVX = (cartX-cartX_1)/dt
        cartA = wahadloShape.body.angle
        arm2_A = wahadlo2Shape.body.angle
        cartVA = (cartA-cartA_1)/dt
        arm2_VA = (arm2_A-arm2_A_1)/dt
        cartX_1=cartX
        cartA_1=cartA
        arm2_A_1 = arm2_A

        x=(cartX, cartVX, cartA, cartVA, arm2_A, arm2_VA)
        #print("x: ", x)

        #aplikujemy prawo sterowania w postaci sieci neuronowej
        xw = [-150, 0, 0, 0, 0, 0]
        e = [(x[0]-xw[0])/100, (x[1]-xw[1])/100, x[2]-xw[2], x[3]-xw[3], x[4]-xw[4], x[5]-xw[5]]
        sE = [sE[0] + abs(e[0]), sE[1] + abs(e[1]), sE[2] + abs(e[2]), sE[3] + abs(e[3]), sE[4] + abs(e[4]),sE[5] + abs(e[5])]
        uL=net.activate(e);
        u=2000*uL[0];

        maxForce=20000;
        if u>maxForce:
            u=maxForce;

        if u<-maxForce:
            u=-maxForce;

        #wytracamy z punktu rownowagi
        if eT<0.03:
            u=10
    
        #tutaj aplikujemy sile zewnetrzna
        body.apply_force_at_world_point((u, 0), body.position)

        #alfaVector=wahadloShape.body.rotation_vector;
        #alfa = np.arctan2(alfaVector[1], alfaVector[0])
        #alfaDegree = np.mod(np.degrees(alfa), 360)
        #print("x_wozka:", shape.body.position[0], "\trotation:", alfaDegree)
        

        ### Draw stuff
        #nie chce miec wyswietlania w trakcie symulacji, zeby dzialalo szybciej
        if isVis==True:
            ### Clear screen
            screen.fill(pygame.Color("white"))
            space.debug_draw(draw_options) 

        ### Update physics
        space.step(dt)
        
        if isVis==True:
            #print("x: ", x)
            print("u: ", u)
            pygame.display.flip()
            clock.tick(fps)

        eT+=dt;

        if eT > 15:
            running = False
            #if isVis==False:
                #print("real time elapsed:", time.time()-start) 
                #print("total time:", eT)
                #print("error sum: ", sE)

    return sE


