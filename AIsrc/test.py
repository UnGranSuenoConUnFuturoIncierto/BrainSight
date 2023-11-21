import trimesh 
from pygltflib import GLTF2 

m = trimesh.creation.uv_sphere()

m.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial())
m.visual.material.baseColorFactor = [0,0,255,100]


m.show() 

m.export(file_obj='blue_sphere.glb')
gltf = GLTF2().load('blue_sphere.glb')
