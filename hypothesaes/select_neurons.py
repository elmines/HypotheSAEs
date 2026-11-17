"""Methods for selecting relevant neurons based on target variables."""

import time
import hashlib
import json
import os
import numpy as np
from typing import List, Optional, Callable, Tuple
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr


def _selection_cache_signature(
    activations: np.ndarray,
    target: np.ndarray,
    method: str,
    parameters: dict,
) -> str:
    digest = hashlib.sha256()
    for array in (activations, target):
        array = np.ascontiguousarray(array)
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(array.tobytes())
    digest.update(method.encode("utf-8"))
    digest.update(json.dumps(parameters, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()


def _load_selection_cache(
    cache_dir: Optional[str],
    signature: str,
) -> Optional[Tuple[List[int], List[float]]]:
    if cache_dir is None:
        return None

    cache_dir = os.fspath(cache_dir)
    metadata_path = os.path.join(cache_dir, ".selection_cache.json")
    indices_path = os.path.join(cache_dir, "indices.npy")
    coefficients_path = os.path.join(cache_dir, "coefficients.npy")
    if not all(os.path.exists(path) for path in (metadata_path, indices_path, coefficients_path)):
        return None

    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        if metadata.get("signature") != signature:
            return None
        indices = np.load(indices_path, allow_pickle=False).astype(int).tolist()
        coefficients = np.load(coefficients_path, allow_pickle=False).astype(float).tolist()
        if len(indices) != len(coefficients):
            return None
        return indices, coefficients
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _save_selection_cache(
    cache_dir: Optional[str],
    signature: str,
    indices: np.ndarray,
    coefficients: np.ndarray,
) -> None:
    if cache_dir is None:
        return

    cache_dir = os.fspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, "indices.npy"), np.asarray(indices, dtype=int))
    np.save(os.path.join(cache_dir, "coefficients.npy"), np.asarray(coefficients, dtype=float))
    with open(os.path.join(cache_dir, ".selection_cache.json"), "w") as f:
        json.dump({"signature": signature}, f)

def select_neurons_lasso(
    activations: np.ndarray,
    target: np.ndarray,
    n_select: int,
    classification: bool = False,
    alpha: Optional[float] = None,
    max_iter: int = 1000,
    verbose: bool = False,
    cache_dir: Optional[str] = None,
) -> Tuple[List[int], List[float]]:
    """
    Select neurons using an L1-regularized linear model (LASSO), which produces sparse coefficient vectors.
    Returns (indices, coefficients) tuple.
    
    Args:
        activations: Neuron activation matrix (n_samples, n_neurons)
        target: Target variable (n_samples,)
        n_select: Number of neurons to select
        classification: Whether this is a classification task
        alpha: LASSO alpha (if None, searches for alpha yielding n_select features)
        max_iter: Maximum iterations for LASSO
        verbose: Whether to print progress of alpha search
    
    Returns:
        Indices of selected neurons and corresponding coefficients
    """
    cache_signature = _selection_cache_signature(
        activations,
        target,
        "lasso",
        {
            "classification": classification,
            "alpha": alpha,
            "max_iter": max_iter,
            "n_select": n_select,
        },
    )
    cached = _load_selection_cache(cache_dir, cache_signature)
    if cached is not None:
        sorted_indices, coefficients = cached
        return sorted_indices[:n_select], coefficients[:n_select]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(activations)
    
    if alpha is not None:
        if verbose:
            print(f"Using provided alpha: {alpha:.2e}")
            
        start_time = time.time()
        if classification:
            model = LogisticRegression(
                l1_ratio=1.0,
                solver='liblinear',
                C=1/alpha,
                max_iter=max_iter
            )
        else:
            model = Lasso(alpha=alpha, max_iter=max_iter)
        
        model.fit(X_scaled, target)
        coef = model.coef_.flatten()
        if verbose:
            print(f"Fitting took {time.time() - start_time:.2f}s")
        
    else:
        # Search for alpha that gives desired number of features
        alpha_low, alpha_high = 1e-6, 1e4
        
        if verbose:
            print(f"{'LASSO iteration':>8} {'L1 Alpha':>10} {'# Features':>10} {'Time (s)':>10}")
            print("-" * 40)
        
        total_start_time = time.time()
        for iteration in range(20):  # Maximum binary search iterations
            iter_start_time = time.time()
            alpha = np.sqrt(alpha_low * alpha_high)
            
            if classification:
                model = LogisticRegression(
                    l1_ratio=1.0,
                    solver='liblinear',
                    C=1/alpha,
                    max_iter=max_iter
                )
            else:
                model = Lasso(alpha=alpha, max_iter=max_iter)
            
            model.fit(X_scaled, target)
            coef = model.coef_.flatten()
            n_nonzero = np.sum(coef != 0)
            iter_time = time.time() - iter_start_time
            
            if verbose:
                print(f"{iteration:8d} {alpha:10.2e} {n_nonzero:10d} {iter_time:10.2f}")
            
            if n_nonzero == n_select:
                break
            elif n_nonzero < n_select:
                alpha_high = alpha
            else:
                alpha_low = alpha
        
        total_time = time.time() - total_start_time
        if verbose and n_nonzero == n_select:
            print(f"\nFound alpha={alpha:.2e} yielding exactly {n_select} features")
            print(f"Total search time: {total_time:.2f}s")
        if n_nonzero != n_select:
            print(f"Warning: Search ended with {n_nonzero} features (target: {n_select})")
    
    # Get top neurons by absolute coefficient value
    sorted_indices = np.argsort(-np.abs(coef))
    sorted_coefficients = coef[sorted_indices]
    _save_selection_cache(
        cache_dir,
        cache_signature,
        sorted_indices,
        sorted_coefficients,
    )

    selected_indices = sorted_indices[:n_select]
    selected_coefs = sorted_coefficients[:n_select]
    return selected_indices.tolist(), selected_coefs.tolist()

def select_neurons_correlation(
    activations: np.ndarray,
    target: np.ndarray,
    n_select: int,
    cache_dir: Optional[str] = None,
    **kwargs
) -> Tuple[List[int], List[float]]:
    """
    Select neurons with highest correlation with target.
    Returns (indices, correlations) tuple.
    
    Args:
        activations: Neuron activation matrix (n_samples, n_neurons)
        target: Target variable (n_samples,)
        n_select: Number of neurons to select
        abs_correlation: Whether to use absolute correlation values
    
    Returns:
        Indices of selected neurons and corresponding correlations
    """
    cache_signature = _selection_cache_signature(
        activations,
        target,
        "correlation",
        {"n_select": n_select},
    )
    cached = _load_selection_cache(cache_dir, cache_signature)
    if cached is not None:
        sorted_indices, coefficients = cached
        return sorted_indices[:n_select], coefficients[:n_select]

    correlations = np.array([
        pearsonr(activations[:, i], target)[0]
        for i in range(activations.shape[1])
    ])
    
    # Sort by absolute correlation but return raw correlations
    sorted_indices = np.argsort(-np.abs(correlations))
    sorted_correlations = correlations[sorted_indices]
    _save_selection_cache(
        cache_dir,
        cache_signature,
        sorted_indices,
        sorted_correlations,
    )

    return (
        sorted_indices[:n_select].tolist(),
        sorted_correlations[:n_select].tolist(),
    )

def select_neurons_separation_score(
    activations: np.ndarray,
    target: np.ndarray,
    n_select: int,
    n_top_activating: int = 100,
    n_zero_activating: Optional[int] = None,
    cache_dir: Optional[str] = None,
    **kwargs
) -> Tuple[List[int], List[float]]:
    """
    Select neurons based on separation between top activations and zero activations.
    Returns (indices, scores) tuple.
    
    Args:
        activations: Neuron activation matrix (n_samples, n_neurons) 
        target: Target variable (n_samples,)
        n_select: Number of neurons to select
        n_top: Number of top activations to use for high activation mean
    
    Returns:
        Indices of selected neurons and corresponding separation scores
    """
    cache_signature = _selection_cache_signature(
        activations,
        target,
        "separation_score",
        {
            "n_select": n_select,
            "n_top_activating": n_top_activating,
            "n_zero_activating": n_zero_activating,
        },
    )
    cached = _load_selection_cache(cache_dir, cache_signature)
    if cached is not None:
        sorted_indices, coefficients = cached
        return sorted_indices[:n_select], coefficients[:n_select]

    scores = []
    for i in range(activations.shape[1]):
        neuron_acts = activations[:, i]
        nonzero_mask = neuron_acts >= 0

        # Find all instances where activation is positive
        nonzero_indices = np.where(nonzero_mask)[0]
        n_positive = len(nonzero_indices)
        sorted_nonzero_indices = nonzero_indices[np.argsort(-neuron_acts[nonzero_indices])]

        # Get mean target value for top n activations, 0 if neuron is dead
        top_mean = np.mean(target[sorted_nonzero_indices[:n_top_activating]]) if n_positive != 0 else 0

        # Print warning message if fewer than n_top_activating positive activations
        if n_positive < n_top_activating:
            print(f"[WARNING] Only found {n_positive} examples with positive activation for neuron {i}, using all available to calculate separation score")
        
        # Get mean target value for zero activations
        zero_indices = neuron_acts == 0
        if n_zero_activating is not None:
            zero_idx = np.where(zero_indices)[0]
            rand_zero_idx = np.random.choice(zero_idx, size=n_zero_activating, replace=False)
            zero_mean = np.mean(target[rand_zero_idx])
        else:
            zero_mean = np.mean(target[zero_indices])
            
        scores.append(top_mean - zero_mean)
    
    scores = np.array(scores)
    sorted_indices = np.argsort(-np.abs(scores))
    sorted_scores = scores[sorted_indices]
    _save_selection_cache(
        cache_dir,
        cache_signature,
        sorted_indices,
        sorted_scores,
    )

    return (
        sorted_indices[:n_select].tolist(),
        sorted_scores[:n_select].tolist(),
    )

def select_neurons_custom(
    activations: np.ndarray,
    target: np.ndarray,
    n_select: int,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    cache_dir: Optional[str] = None,
) -> Tuple[List[int], List[float]]:
    """
    Select neurons using a custom metric function.
    Returns (indices, scores) tuple.
    
    Args:
        activations: Neuron activation matrix (n_samples, n_neurons)
        target: Target variable (n_samples,)
        n_select: Number of neurons to select
        metric_fn: Function that takes (neuron_activations, target) and returns score
    
    Returns:
        Indices of selected neurons and corresponding scores
    """
    metric_name = f"{getattr(metric_fn, '__module__', '')}.{getattr(metric_fn, '__qualname__', repr(metric_fn))}"
    cache_signature = _selection_cache_signature(
        activations,
        target,
        "custom",
        {"n_select": n_select, "metric_fn": metric_name},
    )
    cached = _load_selection_cache(cache_dir, cache_signature)
    if cached is not None:
        sorted_indices, coefficients = cached
        return sorted_indices[:n_select], coefficients[:n_select]

    scores = np.array([
        metric_fn(activations[:, i], target)
        for i in range(activations.shape[1])
    ])
    
    sorted_indices = np.argsort(scores)[::-1]
    sorted_scores = scores[sorted_indices]
    _save_selection_cache(
        cache_dir,
        cache_signature,
        sorted_indices,
        sorted_scores,
    )

    return (
        sorted_indices[:n_select].tolist(),
        sorted_scores[:n_select].tolist(),
    )

def select_neurons(
    activations: np.ndarray,
    target: np.ndarray,
    n_select: int,
    method: str = "lasso",
    classification: bool = False,
    cache_dir: Optional[str] = None,
    **kwargs
) -> Tuple[List[int], List[float]]:
    """
    Select neurons using specified method.
    Returns (indices, scores) tuple where scores are the raw values
    (coefficients for lasso, correlations for correlation method, etc.)
    
    Args:
        activations: Neuron activation matrix (n_samples, n_neurons)
        target: Target variable (n_samples,)
        n_select: Number of neurons to select
        method: One of "lasso", "correlation", or "custom"
        classification: Whether this is a classification task
        cache_dir: Optional directory for caching ranked indices and coefficients/scores
        **kwargs: Additional arguments passed to specific method
    
    Returns:
        Indices of selected neurons and corresponding scores
    """
    # Check if target is multi-class categorical. If so, throw an error.
    if classification and len(np.unique(target)) > 2:
        raise ValueError("classification=True, but the target variable has more than 2 classes. We currently do not support multi-class classification, but you can convert to a one-vs-rest binary classification.")

    if method == "lasso":
        return select_neurons_lasso(
            activations=activations,
            target=target,
            n_select=n_select,
            classification=classification,
            cache_dir=cache_dir,
            **kwargs
        )
    elif method == "correlation":
        return select_neurons_correlation(
            activations=activations,
            target=target,
            n_select=n_select,
            cache_dir=cache_dir,
            **kwargs
        )
    elif method == "separation_score":
        return select_neurons_separation_score(
            activations=activations,
            target=target,
            n_select=n_select,
            cache_dir=cache_dir,
            **kwargs
        )
    elif method == "custom":
        if "metric_fn" not in kwargs:
            raise ValueError("Must provide metric_fn for custom method")
        return select_neurons_custom(
            activations=activations,
            target=target,
            n_select=n_select,
            cache_dir=cache_dir,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown selection method: {method}")