**<details><summary>Click here to install *using Singularity*</summary><p>**
A prebuilt Singularity image for Topaz is available [here](https://singularity-hub.org/collections/2413) and can be installed with:
```
singularity pull shub://nysbc/topaz
```

Then, you can run topaz from within the singularity image with (paths must be changed appropriately):
```
singularity exec --nv -B /mounted_path:/mounted_path /path/to/singularity/container/topaz_latest.sif /usr/local/conda/bin/topaz
```
</p></details>