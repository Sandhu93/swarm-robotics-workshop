import importlib

m = importlib.import_module('01_ants')
m.ANIMATE = False
m.apply_params({'BIAS': 0.9, 'DEPOSIT': 120.0, 'EVAP': 0.01, 'DIFFUSE': 0.02, 'STEP_SIZE': 1.0})
m.reset_simulation(42)
m.walls[:, :] = False

for i in range(120):
    m.step(i)

print('success_after_120=', int((m.found_destination).sum()))
print('max_pher=', float(m.pher.max()))
