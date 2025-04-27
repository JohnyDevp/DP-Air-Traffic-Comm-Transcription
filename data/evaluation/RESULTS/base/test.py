import numpy as np

def wer(ref, hyp):
    ref_words = ref.strip().split()
    hyp_words = hyp.strip().split()
    d = np.zeros((len(ref_words)+1, len(hyp_words)+1), dtype=int)

    for i in range(len(ref_words)+1):
        d[i][0] = i
    for j in range(len(hyp_words)+1):
        d[0][j] = j

    for i in range(1, len(ref_words)+1):
        for j in range(1, len(hyp_words)+1):
            if ref_words[i-1] == hyp_words[j-1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(
                d[i-1][j] + 1,    # deletion
                d[i][j-1] + 1,    # insertion
                d[i-1][j-1] + cost  # substitution
            )

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


# === Example where lowercasing makes WER worse ===
reference = "We discussed AI and its impact"
prediction = "We discussed A I and its impact"

# Original WER (case-sensitive)
original_wer = wer(reference, prediction)
print(f"Original WER (case-sensitive): {original_wer:.3f}")

# Lowercased WER
reference_lower = reference.lower()
prediction_lower = prediction.lower()
lowercased_wer = wer(reference_lower, prediction_lower)
print(f"Lowercased WER: {lowercased_wer:.3f}")

