# Diffusion

This section provides instructions on how to utilize diffusion models within the SLEDGE framework. 

### 1. Training Diffusion
Before training a diffusion model, make sure you have a trained autoencoder checkpoint and latent cache as described in `docs/autoencoder.md`.
You can start a training experiment by running the script:
```bash
cd $SLEDGE_DEVKIT_ROOT/scripts/diffusion/
bash training_diffusion.sh
``` 
Please make sure you added the autoencoder checkpoint path to the bash script. Before training starts, the latent variables will be stored in a Hugging Face dataset format and saved to `$SLEDGE_EXP_ROOT/caches/diffusion_cache`. This format is compatible with the [`accelerate`](https://github.com/huggingface/accelerate) framework and has performance advantages. Read more [here](https://huggingface.co/docs/datasets/about_arrow) if you are interested. Our training pipeline supports [diffusion transformers (DiT)](https://arxiv.org/abs/2212.09748) in four sizes (S, B, L, XL). You can find the experiment folder and checkpoints in `$SLEDGE_EXP_ROOT/exp`. You can also monitor the training with tensorboard. 

### 2. Scenario Synthesis
Given the trained diffusion model, you can generate a set of samples used for driving simulation or the generative metrics. You can set the diffuser checkpoint path and run the following:
```bash
bash scenario_caching_diffusion.sh
```
The samples are stored in `$SLEDGE_EXP_ROOT/caches/scenario_cache` by default. These samples can be simulated in the v0.1 release.
Additional options for route extrapolation by inpainting will be added in a future update.

You can also control generated scenario domains with natural language by overriding `generation_prompt`:

```bash
python $SLEDGE_DEVKIT_ROOT/sledge/script/run_diffusion.py \
  py_func=scenario_caching \
  +diffusion=training_dit_model \
  autoencoder_checkpoint=/path/to/rvae_checkpoint.ckpt \
  diffusion_checkpoint=/path/to/diffusion/checkpoint \
  generation_prompt="mostly boston with some pittsburgh traffic"
```

The prompt parser supports city names / abbreviations (e.g. `boston`, `bos`, `pgh`, `vegas`, `sgp`) and simple weighting phrases such as `8 boston 2 pgh`, `only singapore`, or `mostly vegas`.

### 3. Evaluating Diffusion
Coming soon!
