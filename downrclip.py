
from huggingface_hub import hf_hub_download

for model_name in ['ViT-B-32']: # ['RN50', 'ViT-B-32', 'ViT-L-14']:
    checkpoint_path = hf_hub_download("chendelong/RemoteCLIP", f"RemoteCLIP-{model_name}.pt", cache_dir='checkpoints')
    print(f'{model_name} is downloaded to {checkpoint_path}.')
