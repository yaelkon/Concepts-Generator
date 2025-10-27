import numpy as np

def get_mutuali_exclusive_concepts_indices(concepts, mutuali_exclusive=None):
    if mutuali_exclusive is None:
        return None
    
    mutuali_exclusive_indices = []
    for concept1, concept2 in mutuali_exclusive:
        index1 = concepts.index(concept1)
        index2 = concepts.index(concept2)

        mutuali_exclusive_indices.append((index1, index2))
        
    return mutuali_exclusive_indices


def get_concept_annotations(similarity, concept_prior, concepts, mutuali_exclusive_concepts=None):
    concept_annotations = np.array((similarity[0] > concept_prior).cpu(), dtype=np.int32)

    mutuali_exclusive_concepts_indices = \
        get_mutuali_exclusive_concepts_indices(concepts, mutuali_exclusive_concepts)
    if mutuali_exclusive_concepts_indices is None:
        return concept_annotations
    
    # If mutuali exclusive concepts are annotated, take the dominant one
    for index1, index2 in mutuali_exclusive_concepts_indices:
        if (concept_annotations[index1] * concept_annotations[index2]) == 1:
            weak_index = index2 if similarity[0][index1] > similarity[0][index2] else index1
            concept_annotations[weak_index] = 0
    return concept_annotations
