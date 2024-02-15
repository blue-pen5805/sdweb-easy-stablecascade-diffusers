import launch

try:
  from diffusers import StableCascadeDecoderPipeline
except:
  launch.run_pip(f"install git+https://github.com/kashif/diffusers.git@a3dc21385b7386beb3dab3a9845962ede6765887", "diffusers@wuerstchen-v3")
