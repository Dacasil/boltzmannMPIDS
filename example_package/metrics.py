import itertools


def hamming_distance(vectors):
    # Generate all unique pairs of vectors
    pairs = itertools.combinations(vectors, 2)
    distances = []
    for v1, v2 in pairs:
        # Calculate Hamming distance for the current pair
        distance = sum(1 for a, b in zip(v1, v2) if a != b)
        distances.append(distance)

    mean_distance = sum(distances) / len(distances)
    min_distance = min(distances)

    return distances, mean_distance, min_distance

def get_encoding_vectors(network):
    nv1=network.nv//2
    y = network.w.detach().numpy()
    final_index = -1 if network.bias else len(y)
    encoding_v1 = y[2*nv1:final_index,:nv1]
    encoding_v2 = y[2*nv1:final_index,nv1:2*nv1]
    return encoding_v1, encoding_v2
