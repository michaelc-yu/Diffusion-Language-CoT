#!/bin/bash

echo "Preparing GSM8K..."
python -m scripts.prepare_gsm8k || { echo "GSM8K failed"; exit 1; }

echo "Preparing StrategyQA..."
python -m scripts.prepare_strategyqa || { echo "StrategyQA failed"; exit 1; }

echo "Preparing SVAMP..."
python -m scripts.prepare_svamp || { echo "SVAMP failed"; exit 1; }

echo "All datasets prepared!"
