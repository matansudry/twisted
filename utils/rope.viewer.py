from dm_control import mujoco
from mujoco_viewer import MujocoViewer
import os

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'rope.xml')

physics = mujoco.Physics.from_xml_path(model_path)
model = physics.model.ptr
data = physics.data.ptr

# create the viewer object
viewer = MujocoViewer(model, data)

# simulate and render
for _ in range(10000):
    if viewer.is_alive:
        physics.step()
        viewer.render()
        # print (physics.named.data.geom_xpos)
    else:
        break

# close
viewer.close()