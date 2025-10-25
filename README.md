
Diffusion Language Model CoT

(Hallucination knob can be iterated on top of this)


1: Setup a virtual environment (ideally with python 3.11) and activate it

python3.11 -m venv venv (preferred)
python -m venv venv (can also do this but must use a supported python version)
source venv/bin/activate

2: Install requirements.txt

pip install -r requirements.txt

3: Download and prepare the datasets

chmod +x scripts/prepare_all_datasets.sh
bash scripts/prepare_all_datasets.sh

4: Download the model weights for the pretrained diffusion language models we need

chmod +x scripts/download_plaid.sh
bash scripts/download_plaid.sh

5: Login to wandb

python -m wandb login

6: Modify the train config file in configs/train_base.yaml to desired configs

7: Run training

python -m experiments.run_training

8: Modify the eval config file in configs/eval_base.yaml to desired configs

9: Run evaluation

python -m experiments.run_eval






### Note on Running Scripts

This repo uses Python's **module-based execution** to ensure proper relative imports.

Always run scripts using:
python -m model.train

Avoid running with:
python model/train.py



