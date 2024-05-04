import daam
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image


print("HI")

model = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix")
# model = StableDiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2-base')
model = model.to('cuda')

input_img = Image.open("dataset/1_a.jpg")

from matplotlib import pyplot as plt
import numpy as np

def make_im_subplots(*args):
  fig, ax = plt.subplots(*args)

  for ax_ in ax.flatten():
    ax_.set_xticks([])
    ax_.set_yticks([])

  return fig, ax

prompt = 'Change the axolotl to pikachu'

# Trace through generation
with daam.trace(model) as trc:
  output_image = model(prompt,input_img, num_inference_steps=20).images[0]
  global_heat_map = trc.compute_global_heat_map()

angry_heat_map = global_heat_map.compute_word_heat_map('change')
bald_heat_map = global_heat_map.compute_word_heat_map('axolotl')
papers_heat_map = global_heat_map.compute_word_heat_map('pikachu')

plt.rcParams['figure.figsize'] = (8, 8)
fig, ax = make_im_subplots(2, 2)

# Original image
ax[0, 0].imshow(output_image)

# Angry heat map
angry_heat_map.plot_overlay(output_image, ax=ax[0, 1])

# Bald heat map
bald_heat_map.plot_overlay(output_image, ax=ax[1, 0])

# Papers heat map
papers_heat_map.plot_overlay(output_image, ax=ax[1, 1])

plt.savefig('output_image4.png')
print("Reached")
plt.show()