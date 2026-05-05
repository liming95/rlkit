import numpy as np

try:
    import gymnasium as gym
except ImportError:
    gym = None


class BoxSpace:
    def __init__(self, low, high, shape, seed: int = 0):
        self.low = np.full(shape, low, dtype=np.float32)
        self.high = np.full(shape, high, dtype=np.float32)
        self.shape = shape
        self.rng = np.random.RandomState(seed)

    def seed(self, seed: int):
        self.rng.seed(seed)

    def sample(self):
        return self.rng.uniform(self.low, self.high).astype(np.float32)


class TinyPendulumEnv:
    """Small Pendulum clone used when Gymnasium is not installed."""

    def __init__(self, seed: int = 0):
        self.max_speed = 8.0
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = 10.0
        self.m = 1.0
        self.length = 1.0
        self.rng = np.random.RandomState(seed)
        self.observation_space = BoxSpace(-np.inf, np.inf, (3,), seed)
        self.action_space = BoxSpace(-self.max_torque, self.max_torque, (1,), seed)
        self.state = None

    def seed(self, seed: int):
        self.rng.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.seed(seed)
        high = np.array([np.pi, 1.0], dtype=np.float32)
        self.state = self.rng.uniform(low=-high, high=high).astype(np.float32)
        return self._get_obs()

    def step(self, action):
        theta, theta_dot = self.state
        torque = np.clip(action, -self.max_torque, self.max_torque)[0]
        costs = angle_normalize(theta) ** 2 + 0.1 * theta_dot ** 2 + 0.001 * torque ** 2

        new_theta_dot = theta_dot + (
            3 * self.g / (2 * self.length) * np.sin(theta)
            + 3.0 / (self.m * self.length ** 2) * torque
        ) * self.dt
        new_theta_dot = np.clip(new_theta_dot, -self.max_speed, self.max_speed)
        new_theta = theta + new_theta_dot * self.dt

        self.state = np.array([new_theta, new_theta_dot], dtype=np.float32)
        return self._get_obs(), -float(costs), False, {}

    def close(self):
        pass

    def _get_obs(self):
        theta, theta_dot = self.state
        return np.array([np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)


class TinyContinuousCartPoleEnv:
    """Continuous-action cart-pole for SAC.

    The cart moves left/right under a continuous force. The goal is to keep the
    pole upright and the cart near the center.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, seed: int = 0, render_mode=None):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02
        self.x_threshold = 2.4
        self.theta_threshold_radians = 30 * np.pi / 180
        self.rng = np.random.RandomState(seed)
        self.observation_space = BoxSpace(-np.inf, np.inf, (5,), seed)
        self.action_space = BoxSpace(-1.0, 1.0, (1,), seed)
        self.state = None
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.screen_width = 600
        self.screen_height = 400
        self._warned_missing_pygame = False

    def seed(self, seed: int):
        self.rng.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.seed(seed)
        self.state = self.rng.uniform(low=-0.05, high=0.05, size=(4,)).astype(np.float32)
        if self.render_mode == "human":
            self.render()
        return self._get_obs()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag * float(np.clip(action, -1.0, 1.0)[0])
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        theta_acc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        x_acc = temp - self.polemass_length * theta_acc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * x_acc
        theta = angle_normalize(theta + self.tau * theta_dot)
        theta_dot = theta_dot + self.tau * theta_acc
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        done = bool(
            abs(x) > self.x_threshold
            or abs(theta) > self.theta_threshold_radians
        )
        reward = 1.0 - (
            0.5 * (theta / self.theta_threshold_radians) ** 2
            + 0.1 * (x / self.x_threshold) ** 2
            + 0.001 * force ** 2
        )
        if done:
            reward -= 1.0

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), float(reward), done, {}

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None

    def render(self):
        try:
            import pygame
        except ImportError:
            if not self._warned_missing_pygame:
                print("pygame is not installed; install it to render TinyContinuousCartPole-v0.")
                self._warned_missing_pygame = True
            return None

        if self.state is None:
            return None

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                pygame.display.set_caption("TinyContinuousCartPole-v0")
            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return None

        surface = pygame.Surface((self.screen_width, self.screen_height))
        surface.fill((245, 247, 250))

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        cart_y = int(self.screen_height * 0.62)
        cart_width = 70
        cart_height = 35
        pole_len = int(scale * (2 * self.length))

        x, _, theta, _ = self.state
        cart_x = int(self.screen_width / 2 + x * scale)

        pygame.draw.line(surface, (90, 95, 105), (0, cart_y + cart_height // 2), (self.screen_width, cart_y + cart_height // 2), 2)
        pygame.draw.rect(
            surface,
            (42, 93, 176),
            pygame.Rect(cart_x - cart_width // 2, cart_y - cart_height // 2, cart_width, cart_height),
            border_radius=4,
        )

        axle = (cart_x, cart_y - cart_height // 2)
        pole_tip = (
            int(axle[0] + pole_len * np.sin(theta)),
            int(axle[1] - pole_len * np.cos(theta)),
        )
        pygame.draw.line(surface, (210, 93, 35), axle, pole_tip, 8)
        pygame.draw.circle(surface, (30, 35, 45), axle, 6)
        pygame.draw.circle(surface, (210, 93, 35), pole_tip, 5)

        left_bound = int(self.screen_width / 2 - self.x_threshold * scale)
        right_bound = int(self.screen_width / 2 + self.x_threshold * scale)
        pygame.draw.line(surface, (190, 60, 60), (left_bound, cart_y + 32), (left_bound, cart_y + 56), 3)
        pygame.draw.line(surface, (190, 60, 60), (right_bound, cart_y + 32), (right_bound, cart_y + 56), 3)

        self.screen.blit(surface, (0, 0))
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None

        return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def _get_obs(self):
        x, x_dot, theta, theta_dot = self.state
        return np.array([x, x_dot, np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def make_env(env_name: str, seed: int, render_mode=None):
    if env_name == "TinyContinuousCartPole-v0":
        return TinyContinuousCartPoleEnv(seed=seed, render_mode=render_mode)
    if env_name == "TinyPendulum-v0":
        return TinyPendulumEnv(seed=seed)

    if gym is None:
        print("gymnasium is not installed; using built-in TinyContinuousCartPole-v0 instead.")
        return TinyContinuousCartPoleEnv(seed=seed)

    if render_mode is None:
        env = gym.make(env_name)
    else:
        env = gym.make(env_name, render_mode=render_mode)
    try:
        env.reset(seed=seed)
    except TypeError:
        env.seed(seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)
    return env


def reset_env(env):
    result = env.reset()
    if isinstance(result, tuple):
        obs, _ = result
        return obs
    return result


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        next_obs, reward, terminated, truncated, info = result
        return next_obs, reward, terminated or truncated, info
    next_obs, reward, done, info = result
    return next_obs, reward, done, info


def scale_action(normalized_action, action_low, action_high):
    action = action_low + (normalized_action + 1.0) * 0.5 * (action_high - action_low)
    return np.clip(action, action_low, action_high)


def normalize_action(action, action_low, action_high):
    return (2.0 * (action - action_low) / (action_high - action_low)) - 1.0
