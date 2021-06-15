from PIL import Image
import os
import PIL
import glob

import sys

if len(sys.argv) == 1:
    new_shape = (200, 200)
elif len(sys.argv) == 2:
    new_shape = (int(sys.argv[1]), int(sys.argv[1]))
else:
    new_shape = (int(sys.argv[1]), int(sys.argv[2]))

images_folder = 'minecraft-faces'

images = [file for file in os.listdir(images_folder) if file.endswith(('jpeg', 'png', 'jpg'))]

resized_images_folder = f'{images_folder}-{new_shape[0]}x{new_shape[0]}'

if not os.path.exists(resized_images_folder) or not os.path.isdir(resized_images_folder):
    os.mkdir(resized_images_folder)

processed = 0
for image in images:
    full_path = os.path.join(images_folder, image)
    img = Image.open(full_path)

    resized_full_path = os.path.join(resized_images_folder, image)
    resized_img = img.resize(new_shape)

    l = resized_img.split()

    if len(l) >= 3:
        r, g, b = l[0], l[1], l[2]
        
        resized_img = Image.merge("RGB", (r, g, b))
        resized_img.save(resized_full_path.replace('.png', '.bmp'), optimize=True, quality=40)

        processed += 1

print(f"Resized {processed} out of {len(images)} minecraft faces to {new_shape} ;)")