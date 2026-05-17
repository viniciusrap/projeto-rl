env = gym.make(
        'MountainCarContinuous-v0', 
        render_mode='human')

(obs,_) = env.reset()

for i in range(1000):
    a, _s = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(a)
    env.render()
    if done:
      obs = env.reset()
