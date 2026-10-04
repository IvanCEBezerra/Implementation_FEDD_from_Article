import numpy as np

from fedd.features_extration.features import extract_features
from fedd.features_extration.distances import cosine_distance


vector = np.load(
    "data/generated/linear/abrupt/linear_1_1.npy"
)

window_1 = vector[0:300]
window_2 = vector[300:600]

fv1 = extract_features(window_1)
fv2 = extract_features(window_2)

print("FV1:")
print(fv1)

print("\nFV2:")
print(fv2)

print("\nShapes:")
print(fv1.shape, fv2.shape)

print("\nCosine distance:")
print(cosine_distance(fv1, fv2))