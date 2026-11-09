import numpy as np
import wgpu
from wgpu import gpu
import os
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Ouvrir le GPU
adapter = gpu.request_adapter_sync(power_preference="high-performance")
device = adapter.request_device_sync()

# Préparer les données
W, H = 200, 150
grid = (np.random.random((H, W)) < 0.3).astype(np.uint32)

groups_x = (W + 15) // 16    
groups_y = (H + 15) // 16

numpy_data = grid.flatten()                  # la grille "à plat", comme dans le shader
uniform_data = np.array([W, H], dtype=np.uint32)

# Créer les buffers

# uniform buffer
buffer0 = device.create_buffer_with_data(
    data=uniform_data,
    usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
)

# input data
buffer1 = device.create_buffer_with_data(data=numpy_data, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC)

# for the output
buffer2 = device.create_buffer(
      size=numpy_data.nbytes,
      usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC)

# Bind group layout

from wgpu import ShaderStage, BufferBindingType

bind_group_layout = device.create_bind_group_layout(
    entries=[
        {   # binding 0 : size
            "binding": 0,
            "visibility": ShaderStage.COMPUTE,
            "buffer": {"type": BufferBindingType.uniform},
        },
        {   # binding 1 : grid_in
            "binding": 1,
            "visibility": ShaderStage.COMPUTE,
            "buffer": {"type": BufferBindingType.read_only_storage},
        },
        {   # binding 2 : grid_out
            "binding": 2,
            "visibility": ShaderStage.COMPUTE,
            "buffer": {"type": BufferBindingType.storage},
        },
    ]
)

# Load the source code to execute on the GPU
with open("shader.wgsl") as file:
    shader_source = file.read()

shader_module = device.create_shader_module(code=shader_source)

# A Pipeline may use multiple Bind Group
pipeline_layout = device.create_pipeline_layout(
    bind_group_layouts=[bind_group_layout]
)

pipeline = device.create_compute_pipeline(
    layout=pipeline_layout,
    compute={
        "module": shader_module,
        "entry_point": "main",
    },
)

# Mise en place du Ping Pong

bind_group_a = device.create_bind_group(
    layout=bind_group_layout,
    entries=[
        {   # binding 0 -> size (uniform)
            "binding": 0,
            "resource": {"buffer": buffer0, "offset": 0, "size": buffer0.size},
        },
        {   # binding 1 -> grid_in (lecture)
            "binding": 1,
            "resource": {"buffer": buffer1, "offset": 0, "size": buffer1.size},
        },
        {   # binding 2 -> grid_out (écriture)
            "binding": 2,
            "resource": {"buffer": buffer2, "offset": 0, "size": buffer2.size},
        },
    ],
)

bind_group_b = device.create_bind_group(
    layout=bind_group_layout,
    entries=[
        {"binding": 0, "resource": {"buffer": buffer0, "offset": 0, "size": buffer0.size}},
        {"binding": 1, "resource": {"buffer": buffer2, "offset": 0, "size": buffer2.size}},  # grid_in  = buffer2
        {"binding": 2, "resource": {"buffer": buffer1, "offset": 0, "size": buffer1.size}},  # grid_out = buffer1
    ],
)


# Préparer la figure et l'image de départ
fig, ax = plt.subplots()
im = ax.imshow(grid, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
ax.axis("off")    # pas de graduations autour de l'image


def update(frame):
    # frame = 0, 1, 2, 3... : matplotlib l'augmente de 1 à chaque appel.
    # Ça correspond exactement au nombre de générations déjà calculées.

    # 1. Calculer UNE génération
    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    if frame % 2 == 0:
        compute_pass.set_bind_group(0, bind_group_a)   # lit buffer1, écrit buffer2
        final_buffer = buffer2
    else:
        compute_pass.set_bind_group(0, bind_group_b)   # lit buffer2, écrit buffer1
        final_buffer = buffer1
    compute_pass.dispatch_workgroups(groups_x, groups_y, 1)
    compute_pass.end()
    device.queue.submit([encoder.finish()])

    # 2. Relire le résultat
    result = np.frombuffer(device.queue.read_buffer(final_buffer), dtype=np.uint32).reshape(H, W)

    # 3. Mettre à jour l'image
    im.set_data(result)
    ax.set_title(f"Génération {frame + 1}")
    return [im]


ani = FuncAnimation(fig, update, interval=1000, cache_frame_data=False)
plt.show()