import itertools


def hamming_distance(vectors):
    vectors = vectors.T
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
