import numpy as np
import random

# Parameters
NUM_PARTICLES = 50
MAX_SPEED = 10.0
NEIGHBOR = 35
SEPERATION = 25
AVOID = 75
THRES = 25

F_ALIGN = 0.2
F_AVOID = 1.3 
F_COH = 0.02
F_SEP = 0.3

class Particle:
    def __init__(self, max_x, max_y):
        self.limit_x = max_x
        self.limit_y = max_y

        self.pos = np.array([random.uniform(0, max_x), random.uniform(0, max_y)])
        angle = random.uniform(0, np.pi * 2)
        self.vel = np.array([np.cos(angle), np.sin(angle)])

    def step(self, particles, avoid_points):
        align = np.zeros(2)
        coh = np.zeros(2)
        sep = np.zeros(2)
        total = 0

        # Calculate force for neighbor particles
        for p in particles:
            if p is self: continue

            d = np.linalg.norm(p.pos - self.pos)
            if d < NEIGHBOR:
                total += 1
                align += p.vel
                coh += p.pos

                if d < SEPERATION and d > 0:
                    sep -= (p.pos - self.pos) / d

        if total > 0:
            align = (align / total) - self.vel
            coh = (coh / total) - self.pos

            # Steering force
            self.vel += align * F_ALIGN + coh * F_COH + sep * F_SEP
        
        # Try to avoid specific position (e.g. finger)
        for point in avoid_points:
            d = np.linalg.norm(self.pos - np.float32(point))
            if d < AVOID and d > 0:
                if d < THRES:
                    particles.remove(self)
                    return

                repel =(self.pos - np.float32(point)) / d
                self.vel += repel * F_AVOID

        speed = np.linalg.norm(self.vel)
        if speed > MAX_SPEED:
            self.vel = (self.vel / speed) * MAX_SPEED
        self.pos += self.vel

        # Wrap particles around
        self.pos[0] %= self.limit_x
        self.pos[1] %= self.limit_y

def setup(max_x, max_y):
    return [Particle(max_x, max_y) for _ in range(NUM_PARTICLES)]

def step(particles, avoid_points):
    for p in particles:
        p.step(particles, avoid_points)
    return [(int(p.pos[0]), int(p.pos[1])) for p in particles]
