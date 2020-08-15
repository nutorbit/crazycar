def get_reward_function():
    return {
        "0": "0",
        "theta": "self._speed * math.cos(diffAngle) - self._speed * math.sin(diffAngle)"
    }