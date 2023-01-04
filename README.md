# ASR submission for GARD evaluation (June 2022)

This repo contains the code and scenarios for the evaluation.

## How to run in Docker

### First, install Docker app in your host machine, and run it

### Setting up API token to download the repo

Enter [GitHub Token settins](https://github.com/settings/tokens) and add a token with the access to private repos.
Set `ARMORY_GITHUB_TOKEN` env var with the token's value.

```bash
$  export ARMORY_GITHUB_TOKEN=...
```

### Installing armory

```bash
$ pip install armory-testbed==0.15.3
```

### Obtaining models

Copy the models in your host machine in
```
~/.armory/saved_models
```

List of needed files in ```saved_models```
```
JHUM_adv_denoiser.tar.gz
JHUM_k2_conformer-noam-mmi-att-musan-sa-vgg.tar.gz
JHUM_icefall-conformer.tar.gz
JHUM_k2_icefall_lang_bpe.tar.gz
JHUM_k2_lang_nosp.tar.gz
JHUM_icefall-conformer.yaml
JHUM_snowfall-conformer-noam-mmi-att-musan-sa-vgg-epoch20-avg5.yaml
```


### Running scenarios

```bash
# Run the full evaluation.
$ armory run scenarios/JHUM_icefall_undefended_targeted_entailment.json
# Or just check quickly that it works okay.
$ armory run --check scenarios/JHUM_icefall_undefended_targeted_entailment.json
```
