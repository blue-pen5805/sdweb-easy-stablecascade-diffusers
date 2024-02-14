import launch

modules = [
  "git+https://github.com/kashif/diffusers.git@wuerstchen-v3"
]

try:
  from diffusers import StableCascadeDecoderPipeline
except:
  launch.run_pip(f"install git+https://github.com/kashif/diffusers.git@wuerstchen-v3", "diffusers@wuerstchen-v3")
