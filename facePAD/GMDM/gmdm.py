import numpy as np

def OVL(target, background):
    """
    Calculate the overlap of two distributions.
    
    Parameters:
    target (array-like): Sample values from the target distribution.
    background (array-like): Sample values from the background distribution.
    
    Returns:
    float: Overlap value between the two distributions.
    """
    # Determine the number of bins using Sturges' formula
    nbins = round(1 + np.log2(len(background)))
    
    # Define bin edges
    x = np.linspace(min(min(target), min(background)), max(max(target), max(background)), nbins)
    
    # Estimate probability density function (normalized histogram)
    pdf_t, _ = np.histogram(target, bins=x, density=True)
    pdf_b, _ = np.histogram(background, bins=x, density=True)

    print(pdf_t.shape)
    print(pdf_b.shape)
    
    # Compute overlap
    OVL = np.sum(np.minimum(pdf_t / np.sum(pdf_t), pdf_b / np.sum(pdf_b)))
    print(pdf_t.shape)
    print(pdf_b.shape)
    
    return OVL

# Example usage
# if __name__ == "_main_":
# np.random.seed(42)
# target_samples = np.random.normal(0, 1, 1000)  # Target distribution (normal with mean 0, std 1)

# background_samples = np.random.normal(0.5, 1, 1000)  # Background distribution (normal with mean 0.5, std 1)
# print(target_samples.shape)
# print(background_samples.shape)
# overlap = OVL(target_samples, background_samples)
# print(f"Overlap value: {overlap}")