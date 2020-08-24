import pybullet

from pybullet_envs.bullet import bullet_client

from crazycar.environments.maps import Map
from crazycar.environments.constants import TIMESTEP_SIM, ORIGIN, MAX_STEP
from crazycar.utils import timing


class Environment:
    """
    Crazy Car Environment
    """

    def __init__(self, map_id=1):
        self.p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        self.map_id = map_id
        self.cars = []
        self.position_cars = []

        # variable for tracking
        self.n_collision = 0
        self.step_count = 0
        self.n_reset = 0

        # reset
        self._reset()

    def _reset(self):
        """
        Reset simulation
        """

        self.p.resetSimulation()
        self.p.setTimeStep(TIMESTEP_SIM)
        self.p.setPhysicsEngineParameter(fixedTimeStep=1.0 / 60., numSolverIterations=550, numSubSteps=8)

        # spawn plane
        self.plane_id = self.p.loadURDF("./crazycar/data/plane.urdf")

        # spawn race track
        m = Map(self.p, ORIGIN)
        self.direction_field, self.wall_ids = m.map[self.map_id]()

        # reset common variables
        for i in range(100):
            self.p.stepSimulation()

        self.p.setGravity(0, 0, -10)
        self.step_count = 0
        self.n_collision = 0
        self.n_reset += 1
        self.cars = []

    def restore_cars(self):
        """
        Restore the car without re-insert
        """

        for car_obj, pos in self.position_cars:
            self.insert_car(car_obj, pos)

    def insert_car(self, car_obj, position):
        """
        Insert the car to position

        Args:
            car_obj: car object for create
            position: [x, y, angle (Radians)]
        """

        self.cars.append(
            car_obj(bullet_client=self.p,
                    origin=ORIGIN,
                    carpos=position,
                    plane_id=self.plane_id,
                    direction_field=self.direction_field,
                    wall_ids=self.wall_ids)
        )

        if self.n_reset == 1:
            self.position_cars.append([car_obj, position])

    def get_speed(self):
        """
        Get speed of car in environment

        Returns:
            list (speed for each car)
        """

        res = [car.speed for car in self.cars]
        return res

    def get_obs(self):
        """
        Get observation of car in environment

        Returns:
            list (observation for each car)
        """

        res = [car.get_observation() for car in self.cars]
        return res

    def get_reward(self):
        """
        Get reward of car in environment

        Returns:
            list (reward for each car)
        """

        res = [[car.get_reward()] for car in self.cars]
        return res

    def is_done(self):
        """
        Check whether the environment is done

        Returns:
            True or False
        """

        return [self.step_count > MAX_STEP or any([car.nCollision > 0 for car in self.cars])]

    def get_info(self):
        """
        Get some information

        Returns:
            dictionary
        """

        return {}

    @timing('environment_step', debug=False)
    def step(self, acts):
        """
        apply action to step environment

        Args:
            acts: shape(n, 2), where `n` is a number of cars
        """

        for car, act in zip(self.cars, acts):
            car.apply_action(act)

        self.p.stepSimulation()
        self.speed = acts[:, 0]
        self.step_count += 1

        obs = self.get_obs()
        rew = self.get_reward()
        done = self.is_done()
        info = self.get_info()

        return obs, rew, done, info

    def reset(self):
        """
        Reset the environment

        Returns:
            observation for each car
        """

        self._reset()  # reset environment

        if self.n_reset != 1:  # restore car
            self.restore_cars()

        return self.get_obs()
