import threading
import math
import time


class Ball(threading.Thread):
    def __init__(self, initial_force, angle, interval=0.1):
        super().__init__()
        self.initial_force = initial_force
        self.angle = math.radians(angle)
        self.interval = interval
        self.time_elapsed = 0
        self.position = [0, 0]  # Initial position
        self.velocity = [self.initial_force * math.cos(self.angle),
                         self.initial_force * math.sin(self.angle)]

    def run(self):
        # The forever running loop which runs in the background because of the daemon thread
        while True:
            time.sleep(self.interval)
            self.time_elapsed += self.interval
            self.update_position()

    def update_position(self):
        # Update position based on current velocity
        self.position[0] += self.velocity[0] * self.interval
        self.position[1] += self.velocity[1] * self.interval

        # Check if ball hits the ground
        if self.position[1] <= 0:
            self.position[1] = 0  # Set position to ground level
            self.velocity[1] *= -0.9  # Reverse and reduce velocity due to bounce (elasticity factor 0.9)

        # Update velocity due to gravity
        self.velocity[1] -= 9.81 * self.interval

        # Check if the ball has come to rest
        if abs(self.velocity[1]) < 0.1 and self.position[1] == 0:
            self.velocity = [0, 0]  # Set velocity to zero if the ball has stopped bouncing
    
    def get_position(self):
        return self.position

    def get_velocity(self):
        return self.velocity


if __name__ == "__main__":
    # Above line is required to avoid running the code below when the file is imported while spawning new threads
    initial_force = float(input("Enter initial force: "))
    angle = float(input("Enter launch angle in degrees: "))

    ball1 = Ball(initial_force, angle)
    ball1.daemon = True  # Daemonize the thread

    ball1.start()

    try:
        for _ in range(20):
            time.sleep(2)
            position = ball1.get_position()
            velocity = ball1.get_velocity()
            print(f"Time: {ball1.time_elapsed:.2f}s")
            print(f"Position: ({position[0]:.2f}, {position[1]:.2f})")
            print(f"Velocity: ({velocity[0]:.2f}, {velocity[1]:.2f})")
            print("-----------------------------")
    except KeyboardInterrupt:
        print("Simulation stopped by the user.")
