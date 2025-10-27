import pickle
import os
from re import VERBOSE
import torch
import clip
import yaml
from tqdm import tqdm

from datasets.MetaShift.dataset import MetaShiftDataset
from utils import get_concept_annotations


def main(dataset_name: str, verbose: bool = False):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    with open(f'configs/{dataset_name}.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if config['dataset']['name'] == "MetaShift":
        dataset = MetaShiftDataset(root_dir=config['dataset']['path'], stage=config['dataset']['stage'])
    else:
        raise ValueError(f"Invalid dataset: {config['dataset']['name']}")

    concepts = config['concepts']['names']
    model, preprocess = clip.load("ViT-B/32", device=device)
    text = clip.tokenize(concepts).to(device)
    
    concept_prior = 1 / len(concepts)
    print(f"Concept prior: {concept_prior*100:.2f}%")

    metadata = []
    for item in tqdm(dataset, desc="Processing dataset"):
        image = item['image']
        label = item['label']
        group = item['group']
        image_path = item['image_path']

        image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image).float()
            text_features = model.encode_text(text).float()
            
            # Compute cosine similarity between image and text features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features.float() @ text_features.T.float()).softmax(dim=-1)
        
        concept_annotations = get_concept_annotations(
            similarity=similarity,
            concept_prior=concept_prior,
            concepts=concepts,
            mutuali_exclusive_concepts=config['concepts'].get('mutuali_exclusive', None)
        )

        if verbose:
            print(f"Class: {label}")
            for i in range(len(concepts)):
                print(f"{concepts[i]}: {100 * similarity[0][i]:.2f}%, annotated: {int(concept_annotations[i])}")

        metadata.append({
            'image_path': image_path,
            'label': label,
            'group': group,
            'concept_annotations': concept_annotations,
        })
    
    saving_path = os.path.join(config['dataset']['save_path'], f"{config['dataset']['stage']}_metadata.pkl")
    print(f"Saving metadata to {saving_path}")
    with open(saving_path, 'wb') as f:
        pickle.dump(metadata, f)
    print("Done")


if __name__ == "__main__":
    DATASET_NAME = "MetaShift"
    VERBOSE = False

    main(dataset_name=DATASET_NAME, verbose=VERBOSE)
