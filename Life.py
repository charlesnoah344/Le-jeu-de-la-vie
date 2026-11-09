import numpy as np
import wgpu
from wgpu import gpu

# Ouvrir le GPU
adapter = gpu.request_adapter_sync(power_preference="high-performance")
device = adapter.request_device_sync()

# Préparer les données
W, H = 8, 8                                  # largeur, hauteur
grid = np.zeros((H, W), dtype=np.uint32)     # grille vide, en uint32
grid[1, 2] = 1                               # un planeur
grid[2, 3] = 1
grid[3, 1:4] = 1

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

bind_group = device.create_bind_group(
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

encoder = device.create_command_encoder()       # on prépare une liste de commandes
compute_pass = encoder.begin_compute_pass()
compute_pass.set_pipeline(pipeline)             # quel programme
compute_pass.set_bind_group(0, bind_group)      # avec quelles données
compute_pass.dispatch_workgroups(1, 1, 1)       # combien de workgroups : 1 suffit pour 8×8
compute_pass.end()
device.queue.submit([encoder.finish()])         # on envoie tout au GPU

result = np.frombuffer(device.queue.read_buffer(buffer2), dtype=np.uint32)
print(grid)
print("\n")
print(result.reshape(H, W))
