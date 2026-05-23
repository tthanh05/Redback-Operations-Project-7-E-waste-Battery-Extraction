# Scripting written by Daniel Hernyak 224432628

# Putting this in the blender scripting IDE will allow it to run.
# Make sure there is both the light source and camera still in the environment
# and that the cube is stratched out under the phone to be used as the background

import bpy
import random
import os
from pathlib import Path
import math

# Set output directory
file_dir = Path(bpy.data.filepath).parent
file_name = Path(bpy.data.filepath).stem

output_dir = file_dir / "output_folder"
output_dir.mkdir(exist_ok=True)

print(output_dir)

#Define materials folder
material_folder = file_dir / "material_folder"
material_folder.mkdir(exist_ok=True)
    
#Create a list of all known materials
materials = []
        
for material_file in material_folder.iterdir():
    if material_file.suffix == ".blend":
        materials.append(material_file)
        
#Using the list load all materials into memory once
loaded_materials = []

for material_path in materials:
    with bpy.data.libraries.load(str(material_path), link=False) as (data_from, data_to):
        data_to.materials = data_from.materials

    for mat in data_to.materials:
        if mat is not None:
            loaded_materials.append(mat)
            

# Define all objects in the context
cam = bpy.data.objects['Camera']
target = bpy.data.objects['Cube']
light = bpy.data.objects['Light']

# Remove constraints from camera and light (can be different render to render)
for i in cam.constraints:
        if i.type == 'TRACK_TO':
            cam.constraints.remove(i)
            
for i in light.constraints:
        if i.type == 'TRACK_TO':
            light.constraints.remove(i)
            
            
# Add a new "Track To" constraint to camera and light
track_cam = cam.constraints.new(type='TRACK_TO')
track_cam.target = target

track_light = light.constraints.new(type='TRACK_TO')
track_light.target = target
    
# Configure the axes
# For a standard Blender camera, it points along its local -Z axis with +Y as up
track_cam.track_axis = 'TRACK_NEGATIVE_Z'
track_cam.up_axis = 'UP_Y'

track_light.track_axis = 'TRACK_NEGATIVE_Z'
track_light.up_axis = 'UP_Y'

# Setting boundry and exclusion zones with the boundry cube
# Exclusion zone is defined to be a cylinder around the target object
exclusion_rad_cam = 2
exclusion_rad_light = 5
dist = 0
bounds_cam = 5
bounds_light = 10
lower_height_bound = 15
upper_height_bound = 20

#Create list containing all phone insides within "PhoneInsides" collection
objects_list = bpy.data.collections["PhoneInsides"].objects[:]

for i in objects_list:
    i.hide_render = True

#Create list containing all boders within "Border" collection
border_list = [
    obj for obj in bpy.data.collections["Border"].objects
    if obj.name.lower().endswith("border")
]

for i in border_list:
    i.hide_render = True

print(objects_list)
print(border_list)

for i in range(len(objects_list)):
    #Set current object and relative border to allow for rendering
    objects_list[i].hide_render = False
    border_list[i].hide_render = False

    # Number of renders
    for j in range(10):
        
        # Finding a random material choice via loaded material files
        target.active_material =  random.choice(loaded_materials)
        
        # Finding a coordinate that is within the bounding box and outside the exclusion zone
        while dist <= exclusion_rad_cam:
            x = random.uniform(-bounds_cam, bounds_cam)
            y = random.uniform(-bounds_cam, bounds_cam)
            z = random.uniform(lower_height_bound, upper_height_bound)
            
            dist = math.sqrt((x - target.matrix_world.translation.x)**2 + (y - target.matrix_world.translation.y)**2)

        dist = 0
        cam.location = [x, y, z]
        
        # Finding a coordinate that is within the bounding box and outside the exclusion zone
        while dist <= exclusion_rad_light:
            x = random.uniform(-bounds_light, bounds_light)
            y = random.uniform(-bounds_light, bounds_light)
            z = random.uniform(10, 20)
            
            dist = math.sqrt((x - light.matrix_world.translation.x)**2 + (y - light.matrix_world.translation.y)**2)

        dist = 0
        light.location = [x, y, z]
        
        
        
        # Update scene
        bpy.context.view_layer.update()
        
        # Set output path // using dynamic naming so no changes need to be made to the script when implementing into a blender envirnoment
        safe_name = objects_list[i].name.replace(" ", "_").replace(".", "_")
        bpy.context.scene.render.filepath = str(output_dir / f"render_{safe_name}_{j}.png")
        
        # Render
        bpy.ops.render.render(write_still=True)
        
    #Set current object and border to not allow rendering to prepare for the next phone model
    objects_list[i].hide_render = True
    border_list[i].hide_render = True
