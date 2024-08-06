## Setup
### Environment
Our code is built on the basis of the requirement of the official [Stable Diffusion repository](https://github.com/CompVis/stable-diffusion). To set up their environment, please run:
```
conda env create -f environment/environment.yaml
conda activate ldm
```

If you rather use an existing environment, just run:
```
pip install -r environment/requirements.txt
```

Finally, run:
```
python -m spacy download en_core_web_trf
```

### Hugging Face Diffusers Library
Our code relies also on Hugging Face's [diffusers](https://github.com/huggingface/diffusers) library for downloading the Stable Diffusion model. 


## Inference
If only one prompt is generated
```
python run.py --prompt "a blue boat and a brown cat"  --token_indices [2,3,6,7] --part_of_speech [0,1,0,1] --bbox [10,10,20,20,10,10,20,20,40,40,50,50,40,40,50,50] 
```
Notes:
- PLease input the word index list which you want to pay more attention to for `--token_indices`. 
- Please input the word POS list for `--part_of_sppech`.
- Please input the word layout list for  `--bbox`. If you want use GPT to generate layout, you don't need to input it yet
- If you uncertain words index in the tokenizer sequence or the POS for these words, you don't need to input it yet. In `run.py`, you will be provided with the sequence after passing through the tokenizer or the POS of these words by spacy. Then enter it according to the prompts.
- By setting the parameter `--sd_2_1` and `--run_standard_sd`, you can choose to run the vanilla or our optimized SD model which can be version 2.1 or 1.4.

If you want to use the prompt and retrograde batch generation in T2Icompbench

```
python run_t2i_compbench.py  --data_path "./T2I-CompBench_dataset"  --layout_path "./layout_generation/"
```
Notes:
- Please input the dataset path for `--data_path`
- Please input the layout dataset path for `--layout_path`. If you want to regenrate layout or generating images for other dataset, you can first running `utils/gpt2layout.py` getting all layout for prompts.
```
python utils/gpt2layout.py  --data_path "./T2I-CompBench_dataset"  --save_path "./layout"
```
- Note that our method will download the stable diffusion model `stabilityai/stable-diffusion-2-1-base` or `CompVis/stable-diffusion-v1-4` . If you rather use an existing copy of the model, provide the absolute path using `--model_path`.
- There are more parameters to view the `config.py` file

# Automatic Evaluation

The prompt data we tested comes from [T2I-CompBench](https://github.com/Karine-Huang/T2I-CompBench), and the code we used for quantitative index testing also comes from this paper.