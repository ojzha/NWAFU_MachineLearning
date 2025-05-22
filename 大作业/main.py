def run_experiment(X, y, n_features_list, alpha=1.0, beta=1.0):
    """
    Run feature selection experiment with different numbers of selected features
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        The input data matrix
    y : array-like, shape (n_samples,)
        Ground truth labels
    n_features_list : list
        List of different numbers of features to select
    alpha : float
        Weight parameter for the l2,1-norm regularization
    beta : float
        Weight parameter for the structure preserving constraint
        
    Returns:
    --------
    results : dict
        Dictionary containing the results
    """
    n_clusters = len(np.unique(y))
    results = {
        'n_features': n_features_list,
        'acc': [],
        'nmi': [],
        'baseline_acc': None,
        'baseline_nmi': None
    }
    
    # Baseline: use all features
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(X)
    baseline_acc, baseline_nmi = calculate_metrics(y, pred_labels)
    
    print(f"Baseline (all features): ACC = {baseline_acc:.4f}, NMI = {baseline_nmi:.4f}")
    results['baseline_acc'] = baseline_acc
    results['baseline_nmi'] = baseline_nmi
    
    # Save model for first run to plot convergence
    first_model = None
    
    # Run with different numbers of selected features
    for i, n_features in enumerate(n_features_list):
        print(f"\nTesting with {n_features} features:")
        
        # Feature selection
        fs = StructurePreservingFeatureSelection(
            n_features=n_features, 
            alpha=alpha, 
            beta=beta,
            verbose=True if i == 0 else False,
            reg_lambda=1e-4  # Add small regularization for stability
        )
        X_reduced = fs.fit_transform(X)
        
        # Save the first model for convergence plot
        if i == 0:
            first_model = fs
        
        # Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        pred_labels = kmeans.fit_predict(X_reduced)
        
        # Evaluation
        acc, nmi = calculate_metrics(y, pred_labels)
        results['acc'].append(acc)
        results['nmi'].append(nmi)
        
        print(f"ACC = {acc:.4f}, NMI = {nmi:.4f}")
    
    # Plot convergence curve for the first model if it has convergence history
    if first_model is not None and hasattr(first_model, 'convergence_history_'):
        first_model.plot_convergence()
        print("Convergence curve saved to 'results/convergence_curve.png'")
        print("Convergence data saved to 'results/convergence_curve.csv'")
    
    return results#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Structure Preserving Unsupervised Feature Selection

Implementation of the method proposed in the paper:
"Structure preserving unsupervised feature selection" (2018 Neurocomputing)
by Quanmao Lu, Xuelong Li, and Yongsheng Dong

This implementation follows the original algorithm described in the paper,
using a self-expression model with structure-preserving constraints to
select the most representative features without relying on pseudo cluster labels.

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
import scipy.io as sio
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
import time
import heapq
from scipy import sparse
from munkres import Munkres


class StructurePreservingFeatureSelection:
    """
    Structure Preserving Unsupervised Feature Selection (SPUFS)
    
    This class implements the unsupervised feature selection method based
    on the self-expression model with structure preserving constraints.
    
    Parameters:
    -----------
    n_features : int
        Number of features to select
    alpha : float
        Weight parameter for the l2,1-norm regularization
    beta : float
        Weight parameter for the structure preserving constraint
    max_iter : int
        Maximum number of iterations
    tol : float
        Tolerance for stopping criterion
    epsilon : float
        Small constant to avoid division by zero
    sigma : float or None
        Bandwidth parameter for the RBF kernel. If None, automatically determined
    verbose : bool
        Whether to print progress messages
    """
    
    def __init__(self, n_features=100, alpha=1.0, beta=1.0, max_iter=50, 
                 tol=1e-4, epsilon=1e-6, sigma=None, verbose=True, reg_lambda=1e-6):
        self.n_features = n_features
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.tol = tol
        self.epsilon = epsilon
        self.sigma = sigma
        self.verbose = verbose
        self.reg_lambda = reg_lambda  # Regularization parameter for numerical stability
        
        # Results to be computed during fit
        self.W = None  # Feature selection matrix
        self.selected_features_ = None  # Indices of selected features
        self.feature_scores_ = None  # Importance score for each feature
        self.n_iter_ = 0  # Number of iterations run
        
        # Results to be computed during fit
        self.W = None  # Feature selection matrix
        self.selected_features_ = None  # Indices of selected features
        self.feature_scores_ = None  # Importance score for each feature
        self.n_iter_ = 0  # Number of iterations run
        
    def fit(self, X):
        """
        Fit the SPUFS model to find the most representative features
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data matrix
            
        Returns:
        --------
        self : object
            Returns self
        """
        start_time = time.time()
        
        # Get data dimensions
        n_samples, n_features = X.shape
        
        if self.verbose:
            print(f"Running Structure Preserving Unsupervised Feature Selection...")
            print(f"Dataset shape: {X.shape}")
            print(f"Parameters: alpha={self.alpha}, beta={self.beta}, n_features={self.n_features}")
        
        # Step 1: Compute similarity matrix S using RBF kernel
        if self.verbose:
            print("Computing similarity matrix S...")
        
        if self.sigma is None:
            # Automatically determine sigma as the average distance
            # between samples, which is a common heuristic
            dists = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)
            dists = np.sqrt(np.maximum(dists, 0))
            self.sigma = np.mean(dists)
            if self.verbose:
                print(f"Automatically determined sigma = {self.sigma:.4f}")
        
        # Compute similarity matrix using RBF kernel
        S = rbf_kernel(X, gamma=1.0/(2*self.sigma**2))
        
        # Step 2: Compute Laplacian matrix L
        if self.verbose:
            print("Computing Laplacian matrix L...")
        
        # Compute diagonal degree matrix D
        D = np.diag(np.sum(S, axis=1))
        
        # Compute Laplacian matrix L = D - S
        L = D - S
        
        # Step 3: Initialize variables
        if self.verbose:
            print("Initializing variables...")
            
        # Initialize W randomly
        W = np.random.randn(n_features, n_features)
        
        # Initialize diagonal matrix Q (called Lambda in the algorithm)
        Q = np.eye(n_features)
        
        # Pre-compute some matrices to speed up computation
        XTX = X.T @ X
        XTLX = X.T @ L @ X
        
        # Step 4: Alternative optimization
        if self.verbose:
            print("Starting alternative optimization...")
        
        prev_W = W.copy()
        converged = False
        
        # Track convergence
        self.convergence_history_ = {
            'iteration': [],
            'w_diff': [],
            'obj_value': []
        }
        
        for iter_num in range(1, self.max_iter + 1):
            # Update W with fixed Q
            # Add small regularization to prevent singularity
            reg_matrix = XTX + self.beta * XTLX + self.alpha * Q + self.reg_lambda * np.eye(n_features)
            
            # Use pseudo-inverse for better numerical stability
            try:
                # Try direct inverse first
                W = np.linalg.inv(reg_matrix) @ XTX
            except np.linalg.LinAlgError:
                # If that fails, try pseudo-inverse
                if self.verbose:
                    print("Warning: Matrix is singular, using pseudo-inverse.")
                W = np.linalg.pinv(reg_matrix) @ XTX
            
            # Update Q with fixed W
            diag_Q = []
            for i in range(n_features):
                w_i_norm = np.linalg.norm(W[i, :], ord=2)
                diag_Q.append(1.0 / (2 * np.sqrt(w_i_norm**2 + self.epsilon)))
            
            Q = np.diag(diag_Q)
            
            # Calculate objective function value for convergence monitoring
            # ||X - XW||_F^2 + α||W||_{2,1} + (β/2)∑_{i,j} ||W^T x_i - W^T x_j||_2^2 S_{ij}
            reconstruction_error = np.linalg.norm(X - X @ W, 'fro')**2
            sparsity_term = self.alpha * np.sum([np.linalg.norm(W[i, :], ord=2) for i in range(n_features)])
            structure_term = self.beta * np.trace(W.T @ XTLX @ W)
            obj_value = reconstruction_error + sparsity_term + structure_term
            
            # Check convergence
            W_diff = np.linalg.norm(W - prev_W, ord='fro')
            
            # Store convergence data
            self.convergence_history_['iteration'].append(iter_num)
            self.convergence_history_['w_diff'].append(W_diff)
            self.convergence_history_['obj_value'].append(obj_value)
            
            if W_diff < self.tol:
                converged = True
                break
            
            prev_W = W.copy()
            
            if self.verbose and iter_num % 5 == 0:
                print(f"Iteration {iter_num}, W difference: {W_diff:.6f}, Objective value: {obj_value:.6f}")
        
        self.n_iter_ = iter_num
        if self.verbose:
            if converged:
                print(f"Converged after {self.n_iter_} iterations.")
            else:
                print(f"Maximum iterations ({self.max_iter}) reached without convergence.")
        
        # Step 5: Compute feature scores and select top features
        self.W = W
        
        # Compute feature importance scores as l2-norm of each row in W
        self.feature_scores_ = np.linalg.norm(W, axis=1)
        
        # Select top n_features based on feature scores
        self.selected_features_ = heapq.nlargest(self.n_features, 
                                                 range(len(self.feature_scores_)), 
                                                 self.feature_scores_.__getitem__)
        
        if self.verbose:
            elapsed_time = time.time() - start_time
            print(f"Feature selection completed in {elapsed_time:.2f} seconds.")
            print(f"Selected {self.n_features} features.")
        
        return self
    
    def transform(self, X):
        """
        Reduce X to the selected features
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data matrix
            
        Returns:
        --------
        X_reduced : array, shape (n_samples, n_selected_features)
            The reduced data matrix with only the selected features
        """
        if self.selected_features_ is None:
            raise ValueError("The model has not been fitted yet. Call 'fit' first.")
        
        return X[:, self.selected_features_]
    
    def fit_transform(self, X):
        """
        Fit the model with X and apply the dimensionality reduction
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            New data
            
        Returns:
        --------
        X_reduced : array, shape (n_samples, n_selected_features)
            The reduced data matrix with only the selected features
        """
        self.fit(X)
        return self.transform(X)
    
    def plot_feature_scores(self, top_n=50):
        """
        Plot the feature importance scores
        
        Parameters:
        -----------
        top_n : int
            Number of top features to display in the plot
        """
        if self.feature_scores_ is None:
            raise ValueError("The model has not been fitted yet. Call 'fit' first.")
        
        # Get indices of top n features by scores
        top_indices = heapq.nlargest(top_n, range(len(self.feature_scores_)), 
                                      self.feature_scores_.__getitem__)
        top_scores = self.feature_scores_[top_indices]
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(top_indices)), top_scores)
        plt.xlabel('Feature Index')
        plt.ylabel('Feature Importance Score')
        plt.title(f'Top {top_n} Feature Importance Scores')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('results/feature_scores.png')
        plt.close()
        
    def plot_convergence(self, save_path='results/convergence_curve.png'):
        """
        Plot the convergence curve showing how the objective function and W difference
        change during the optimization process
        
        Parameters:
        -----------
        save_path : str
            Path where to save the convergence plot
        """
        if not hasattr(self, 'convergence_history_'):
            raise ValueError("The model has not been fitted yet or convergence history was not recorded.")
        
        iterations = self.convergence_history_['iteration']
        w_diffs = self.convergence_history_['w_diff']
        obj_values = self.convergence_history_['obj_value']
        
        # Safety check - ensure we don't have any extreme outliers that would cause plotting issues
        if len(obj_values) > 0:
            # Replace any extreme outliers with more reasonable values
            max_reasonable_value = 1e10
            obj_values = np.array(obj_values)
            if np.any(obj_values > max_reasonable_value):
                print(f"Warning: Found extremely large objective values (max: {np.max(obj_values)}). Clipping for visualization.")
                obj_values = np.clip(obj_values, None, max_reasonable_value)
        
        # Create two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot W difference
        ax1.plot(iterations, w_diffs, 'b-o', linewidth=2, markersize=4, alpha=0.7)
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('W Difference (Frobenius Norm)', fontsize=12)
        ax1.set_title('Convergence of W Matrix', fontsize=14)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.tick_params(axis='both', which='major', labelsize=10)
        
        # Highlight convergence threshold
        if hasattr(self, 'tol'):
            ax1.axhline(y=self.tol, color='r', linestyle='--', alpha=0.5, label=f'Threshold = {self.tol}')
            ax1.legend(fontsize=10)
        
        # Plot objective function value
        ax2.plot(iterations, obj_values, 'g-o', linewidth=2, markersize=4, alpha=0.7)
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Objective Function Value', fontsize=12)
        ax2.set_title('Convergence of Objective Function', fontsize=14)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.tick_params(axis='both', which='major', labelsize=10)
        
        # Use log scale for objective function if values span multiple orders of magnitude
        if len(obj_values) > 0 and max(obj_values) / (min(obj_values) + 1e-10) > 100:
            ax2.set_yscale('log')
            ax2.set_ylabel('Objective Function Value (log scale)', fontsize=12)
        
        # Safe annotations that won't stretch the figure
        if len(w_diffs) > 0:
            ax1.annotate(f'Final: {w_diffs[-1]:.6f}', 
                        xy=(iterations[-1], w_diffs[-1]),
                        xytext=(iterations[-1] - min(5, iterations[-1]//2), w_diffs[-1] * 1.1), 
                        arrowprops=dict(arrowstyle='->'))
        
        if len(obj_values) > 0:
            ax2.annotate(f'Final: {obj_values[-1]:.2f}', 
                        xy=(iterations[-1], obj_values[-1]),
                        xytext=(iterations[-1] - min(5, iterations[-1]//2), obj_values[-1] * 1.1), 
                        arrowprops=dict(arrowstyle='->'))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save convergence data to CSV
        csv_path = save_path.replace('.png', '.csv')
        import pandas as pd
        df = pd.DataFrame({
            'Iteration': iterations,
            'W_Difference': w_diffs,
            'Objective_Value': obj_values
        })
        df.to_csv(csv_path, index=False)
        
        return df


def load_mnist_data(path="./mnist.mat"):
    """
    Load MNIST dataset from .mat file
    
    Parameters:
    -----------
    path : str
        Path to the MNIST .mat file
        
    Returns:
    --------
    X : array, shape (n_samples, n_features)
        Data matrix
    y : array, shape (n_samples,)
        Ground truth labels
    """
    try:
        data = sio.loadmat(path)
        X = data['X']  # Data matrix
        y = data['Y'].ravel()  # Labels
        if y.min() == 1:  # If labels start from 1, convert to 0-based
            y = y - 1
        return X, y
    except Exception as e:
        print(f"Error loading MNIST data: {e}")
        print("Please make sure the MNIST.mat file contains 'X' and 'Y' variables.")
        raise


def best_map(true_labels, pred_labels):
    """
    Find the best mapping between true labels and predicted labels
    
    Parameters:
    -----------
    true_labels : array-like
        Ground truth labels
    pred_labels : array-like
        Predicted cluster labels
        
    Returns:
    --------
    new_pred : array
        Predicted labels after remapping
    """
    # Get unique labels
    L1 = np.unique(true_labels)
    L2 = np.unique(pred_labels)
    n_class1 = len(L1)
    n_class2 = len(L2)
    n_class = max(n_class1, n_class2)
    
    # Initialize cost matrix
    G = np.zeros((n_class, n_class))
    
    for i in range(n_class1):
        idx1 = (true_labels == L1[i])
        for j in range(n_class2):
            idx2 = (pred_labels == L2[j])
            G[i, j] = np.sum(idx1 & idx2)
    
    # Use Hungarian algorithm to find the best matching
    m = Munkres()
    index = m.compute(-G.T)
    
    # Remap labels
    new_pred = np.zeros(pred_labels.shape)
    for i, j in index:
        new_pred[pred_labels == L2[i]] = L1[j]
    
    return new_pred


def calculate_metrics(true_labels, pred_labels):
    """
    Calculate clustering metrics (ACC and NMI)
    
    Parameters:
    -----------
    true_labels : array-like
        Ground truth labels
    pred_labels : array-like
        Predicted cluster labels
        
    Returns:
    --------
    acc : float
        Clustering accuracy
    nmi : float
        Normalized Mutual Information
    """
    # Calculate Accuracy (ACC)
    mapped_pred = best_map(true_labels, pred_labels)
    acc = np.sum(mapped_pred == true_labels) / len(true_labels)
    
    # Calculate Normalized Mutual Information (NMI)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    
    return acc, nmi


def run_experiment(X, y, n_features_list, alpha=1.0, beta=1.0):
    """
    Run feature selection experiment with different numbers of selected features
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        The input data matrix
    y : array-like, shape (n_samples,)
        Ground truth labels
    n_features_list : list
        List of different numbers of features to select
    alpha : float
        Weight parameter for the l2,1-norm regularization
    beta : float
        Weight parameter for the structure preserving constraint
        
    Returns:
    --------
    results : dict
        Dictionary containing the results
    """
    n_clusters = len(np.unique(y))
    results = {
        'n_features': n_features_list,
        'acc': [],
        'nmi': [],
        'baseline_acc': None,
        'baseline_nmi': None
    }
    
    # Baseline: use all features
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(X)
    baseline_acc, baseline_nmi = calculate_metrics(y, pred_labels)
    
    print(f"Baseline (all features): ACC = {baseline_acc:.4f}, NMI = {baseline_nmi:.4f}")
    results['baseline_acc'] = baseline_acc
    results['baseline_nmi'] = baseline_nmi
    
    # Run with different numbers of selected features
    for n_features in n_features_list:
        print(f"\nTesting with {n_features} features:")
        
        # Feature selection
        fs = StructurePreservingFeatureSelection(
            n_features=n_features, 
            alpha=alpha, 
            beta=beta,
            verbose=False,
            reg_lambda=1e-4  # Add small regularization for stability
        )
        X_reduced = fs.fit_transform(X)
        
        # Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        pred_labels = kmeans.fit_predict(X_reduced)
        
        # Evaluation
        acc, nmi = calculate_metrics(y, pred_labels)
        results['acc'].append(acc)
        results['nmi'].append(nmi)
        
        print(f"ACC = {acc:.4f}, NMI = {nmi:.4f}")
    
    return results


def parameter_sensitivity_analysis(X, y, alphas, betas, n_features=100):
    """
    Analyze the sensitivity of the method to alpha and beta parameters
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        The input data matrix
    y : array-like, shape (n_samples,)
        Ground truth labels
    alphas : list
        List of alpha values to test
    betas : list
        List of beta values to test
    n_features : int
        Number of features to select
        
    Returns:
    --------
    results : dict
        Dictionary containing the results
    """
    n_clusters = len(np.unique(y))
    results = {
        'alphas': alphas,
        'betas': betas,
        'acc': np.zeros((len(alphas), len(betas))),
        'nmi': np.zeros((len(alphas), len(betas)))
    }
    
    # Track the best model for convergence plot
    best_acc = -1
    best_model = None
    
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            print(f"\nTesting with alpha={alpha}, beta={beta}:")
            
            # Feature selection
            fs = StructurePreservingFeatureSelection(
                n_features=n_features, 
                alpha=alpha, 
                beta=beta,
                verbose=False,
                reg_lambda=1e-4  # Add small regularization for stability
            )
            X_reduced = fs.fit_transform(X)
            
            # Clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            pred_labels = kmeans.fit_predict(X_reduced)
            
            # Evaluation
            acc, nmi = calculate_metrics(y, pred_labels)
            results['acc'][i, j] = acc
            results['nmi'][i, j] = nmi
            
            # Track best model for convergence plot
            if acc > best_acc:
                best_acc = acc
                best_model = fs
            
            print(f"ACC = {acc:.4f}, NMI = {nmi:.4f}")
    
    # If we have a best model, save its convergence data
    if best_model is not None:
        i_best, j_best = np.unravel_index(np.argmax(results['acc']), results['acc'].shape)
        alpha_best = alphas[i_best]
        beta_best = betas[j_best]
        
        if hasattr(best_model, 'convergence_history_'):
            best_model.plot_convergence(save_path=f'results/convergence_best_params_a{alpha_best}_b{beta_best}.png')
            print(f"Convergence curve for best parameters (alpha={alpha_best}, beta={beta_best}) "
                  f"saved to 'results/convergence_best_params_a{alpha_best}_b{beta_best}.png'")
    
    return results


def plot_parameter_sensitivity(results):
    """
    Plot the parameter sensitivity analysis results with enhanced visuals
    
    Parameters:
    -----------
    results : dict
        Results from parameter_sensitivity_analysis
    """
    alphas = results['alphas']
    betas = results['betas']
    acc_matrix = results['acc']
    nmi_matrix = results['nmi']
    
    log_alphas = np.log10(np.array(alphas))
    log_betas = np.log10(np.array(betas))
    
    # Create meshgrid for surface plot
    alpha_grid, beta_grid = np.meshgrid(log_alphas, log_betas)
    
    # Enhanced 3D visualization
    # Plot ACC
    fig = plt.figure(figsize=(15, 10))
    
    # ACC plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(alpha_grid, beta_grid, acc_matrix.T, cmap='viridis', 
                            antialiased=True, edgecolor='none', alpha=0.8,
                            rstride=1, cstride=1, linewidth=0)
    
    # Add wireframe for better 3D perception
    ax1.plot_wireframe(alpha_grid, beta_grid, acc_matrix.T, color='black', alpha=0.1, linewidth=0.5)
    
    # Add contour plot at the bottom for better reference
    offset = np.min(acc_matrix) - 0.1 * (np.max(acc_matrix) - np.min(acc_matrix))
    ax1.contourf(alpha_grid, beta_grid, acc_matrix.T, zdir='z', offset=offset, cmap='viridis', alpha=0.5)
    
    ax1.set_xlabel('log10(α)', fontsize=14, labelpad=10)
    ax1.set_ylabel('log10(β)', fontsize=14, labelpad=10)
    ax1.set_zlabel('ACC', fontsize=14, labelpad=10)
    ax1.set_title('Clustering Accuracy vs. Parameters', fontsize=16)
    
    # Add annotation for best parameter combination
    i_max, j_max = np.unravel_index(np.argmax(acc_matrix), acc_matrix.shape)
    best_alpha = alphas[i_max]
    best_beta = betas[j_max]
    best_acc = acc_matrix[i_max, j_max]
    ax1.text(log_alphas[i_max], log_betas[j_max], best_acc, 
            f'Best: α={best_alpha}, β={best_beta}\nACC={best_acc:.4f}',
            color='red', fontsize=12, ha='center', va='bottom')
    
    # Highlight the best point
    ax1.scatter([log_alphas[i_max]], [log_betas[j_max]], [best_acc], 
               color='red', s=100, marker='*', edgecolor='black', linewidth=1)
    
    # Adjust axes limits for better view
    ax1.set_xlim(min(log_alphas), max(log_alphas))
    ax1.set_ylim(min(log_betas), max(log_betas))
    z_range = np.max(acc_matrix) - np.min(acc_matrix)
    ax1.set_zlim(offset, np.max(acc_matrix) + 0.1 * z_range)
    
    # Custom colorbar
    cbar1 = fig.colorbar(surf1, ax=ax1, shrink=0.6, aspect=10, pad=0.1)
    cbar1.set_label('ACC', fontsize=12)
    
    # Rotate the view for better 3D perception
    ax1.view_init(elev=30, azim=225)
    
    # NMI plot
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(alpha_grid, beta_grid, nmi_matrix.T, cmap='plasma', 
                            antialiased=True, edgecolor='none', alpha=0.8,
                            rstride=1, cstride=1, linewidth=0)
    
    # Add wireframe for better 3D perception
    ax2.plot_wireframe(alpha_grid, beta_grid, nmi_matrix.T, color='black', alpha=0.1, linewidth=0.5)
    
    # Add contour plot at the bottom for better reference
    offset = np.min(nmi_matrix) - 0.1 * (np.max(nmi_matrix) - np.min(nmi_matrix))
    ax2.contourf(alpha_grid, beta_grid, nmi_matrix.T, zdir='z', offset=offset, cmap='plasma', alpha=0.5)
    
    ax2.set_xlabel('log10(α)', fontsize=14, labelpad=10)
    ax2.set_ylabel('log10(β)', fontsize=14, labelpad=10)
    ax2.set_zlabel('NMI', fontsize=14, labelpad=10)
    ax2.set_title('Normalized Mutual Information vs. Parameters', fontsize=16)
    
    # Add annotation for best parameter combination
    i_max, j_max = np.unravel_index(np.argmax(nmi_matrix), nmi_matrix.shape)
    best_alpha = alphas[i_max]
    best_beta = betas[j_max]
    best_nmi = nmi_matrix[i_max, j_max]
    ax2.text(log_alphas[i_max], log_betas[j_max], best_nmi, 
            f'Best: α={best_alpha}, β={best_beta}\nNMI={best_nmi:.4f}',
            color='red', fontsize=12, ha='center', va='bottom')
    
    # Highlight the best point
    ax2.scatter([log_alphas[i_max]], [log_betas[j_max]], [best_nmi], 
               color='red', s=100, marker='*', edgecolor='black', linewidth=1)
    
    # Adjust axes limits for better view
    ax2.set_xlim(min(log_alphas), max(log_alphas))
    ax2.set_ylim(min(log_betas), max(log_betas))
    z_range = np.max(nmi_matrix) - np.min(nmi_matrix)
    ax2.set_zlim(offset, np.max(nmi_matrix) + 0.1 * z_range)
    
    # Custom colorbar
    cbar2 = fig.colorbar(surf2, ax=ax2, shrink=0.6, aspect=10, pad=0.1)
    cbar2.set_label('NMI', fontsize=12)
    
    # Rotate the view for better 3D perception
    ax2.view_init(elev=30, azim=225)
    
    plt.tight_layout()
    plt.savefig('results/parameter_sensitivity_3d.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create additional 2D heatmap plots for better visualization
    plt.figure(figsize=(15, 6))
    
    # ACC heatmap
    plt.subplot(1, 2, 1)
    im1 = plt.imshow(acc_matrix, cmap='viridis', aspect='auto', 
               extent=[min(log_betas), max(log_betas), min(log_alphas), max(log_alphas)],
               origin='lower')
    plt.colorbar(im1, label='ACC')
    plt.xlabel('log10(β)', fontsize=12)
    plt.ylabel('log10(α)', fontsize=12)
    plt.title('ACC vs. Parameters (Heatmap)', fontsize=14)
    
    # Add text annotations to the heatmap
    for i in range(len(alphas)):
        for j in range(len(betas)):
            plt.text(log_betas[j], log_alphas[i], f'{acc_matrix[i,j]:.3f}', 
                    ha='center', va='center', color='white' if acc_matrix[i,j] < 0.7*np.max(acc_matrix) else 'black',
                    fontsize=8)
    
    # NMI heatmap
    plt.subplot(1, 2, 2)
    im2 = plt.imshow(nmi_matrix, cmap='plasma', aspect='auto',
               extent=[min(log_betas), max(log_betas), min(log_alphas), max(log_alphas)],
               origin='lower')
    plt.colorbar(im2, label='NMI')
    plt.xlabel('log10(β)', fontsize=12)
    plt.ylabel('log10(α)', fontsize=12)
    plt.title('NMI vs. Parameters (Heatmap)', fontsize=14)
    
    # Add text annotations to the heatmap
    for i in range(len(alphas)):
        for j in range(len(betas)):
            plt.text(log_betas[j], log_alphas[i], f'{nmi_matrix[i,j]:.3f}', 
                    ha='center', va='center', color='white' if nmi_matrix[i,j] < 0.7*np.max(nmi_matrix) else 'black',
                    fontsize=8)
    
    plt.tight_layout()
    plt.savefig('results/parameter_sensitivity_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate parameter tables for Origin
    # Table 1: Alpha-Beta-ACC table for 3D surface plot in Origin
    with open('results/param_acc_origin.txt', 'w') as f:
        # Header
        f.write("Alpha\tBeta\tACC\n")
        
        # Data in format suitable for Origin 3D plot
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                f.write(f"{alpha}\t{beta}\t{acc_matrix[i,j]:.6f}\n")
    
    # Table 2: Alpha-Beta-NMI table for 3D surface plot in Origin
    with open('results/param_nmi_origin.txt', 'w') as f:
        # Header
        f.write("Alpha\tBeta\tNMI\n")
        
        # Data in format suitable for Origin 3D plot
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                f.write(f"{alpha}\t{beta}\t{nmi_matrix[i,j]:.6f}\n")
    
    # Table 3: Matrix format ACC for easier import in Origin
    with open('results/param_acc_matrix.txt', 'w') as f:
        # Header with beta values
        f.write("Alpha/Beta\t" + "\t".join([f"{beta}" for beta in betas]) + "\n")
        
        # Each row starts with alpha value followed by ACC values
        for i, alpha in enumerate(alphas):
            row = [f"{alpha}"] + [f"{acc_matrix[i,j]:.6f}" for j in range(len(betas))]
            f.write("\t".join(row) + "\n")
    
    # Table 4: Matrix format NMI for easier import in Origin
    with open('results/param_nmi_matrix.txt', 'w') as f:
        # Header with beta values
        f.write("Alpha/Beta\t" + "\t".join([f"{beta}" for beta in betas]) + "\n")
        
        # Each row starts with alpha value followed by NMI values
        for i, alpha in enumerate(alphas):
            row = [f"{alpha}"] + [f"{nmi_matrix[i,j]:.6f}" for j in range(len(betas))]
            f.write("\t".join(row) + "\n")


def plot_results(results):
    """
    Plot the results of the feature selection experiment
    
    Parameters:
    -----------
    results : dict
        Results from run_experiment
    """
    n_features = results['n_features']
    acc = results['acc']
    nmi = results['nmi']
    baseline_acc = results['baseline_acc']
    baseline_nmi = results['baseline_nmi']
    
    plt.figure(figsize=(12, 5))
    
    # ACC plot
    plt.subplot(1, 2, 1)
    plt.plot(n_features, acc, 'o-', label='SPUFS')
    plt.axhline(y=baseline_acc, color='r', linestyle='--', label='All Features')
    plt.xlabel('Number of Selected Features')
    plt.ylabel('ACC')
    plt.title('Clustering Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # NMI plot
    plt.subplot(1, 2, 2)
    plt.plot(n_features, nmi, 'o-', label='SPUFS')
    plt.axhline(y=baseline_nmi, color='r', linestyle='--', label='All Features')
    plt.xlabel('Number of Selected Features')
    plt.ylabel('NMI')
    plt.title('Normalized Mutual Information')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/feature_selection_results.png')
    plt.close()


def main():
    """Main function to run the experiments"""
    
    # Create output directory for figures if it doesn't exist
    import os
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    X, y = load_mnist_data()
    print(f"Dataset loaded: {X.shape} samples with {X.shape[1]} features")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Feature selection with different numbers of features
    print("\n=== Running feature selection experiment ===")
    n_features_list = [50, 100, 150, 200, 250, 300]
    # Use default max_iter value
    results = run_experiment(X, y, n_features_list, alpha=1.0, beta=1.0)
    
    # Plot results
    plot_results(results)
    print("Results saved to 'results/feature_selection_results.png'")
    
    # Parameter sensitivity analysis
    print("\n=== Running parameter sensitivity analysis ===")
    # Use more values for smoother 3D plots
    alphas = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    betas = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    param_results = parameter_sensitivity_analysis(X, y, alphas, betas, n_features=100)
    
    # Plot parameter sensitivity results
    plot_parameter_sensitivity(param_results)
    print("Parameter sensitivity results saved to 'results/parameter_sensitivity_3d.png' and 'results/parameter_sensitivity_heatmap.png'")
    print("Parameter tables for Origin saved to 'results/param_acc_origin.txt', 'results/param_nmi_origin.txt', 'results/param_acc_matrix.txt', and 'results/param_nmi_matrix.txt'")
    
    # Save numerical results to CSV files
    import pandas as pd
    
    # Feature selection results
    df_results = pd.DataFrame({
        'n_features': n_features_list,
        'ACC': results['acc'],
        'NMI': results['nmi'],
        'baseline_ACC': [results['baseline_acc']] * len(n_features_list),
        'baseline_NMI': [results['baseline_nmi']] * len(n_features_list)
    })
    df_results.to_csv('results/feature_selection_results.csv', index=False)
    
    # Parameter sensitivity results
    df_acc = pd.DataFrame(param_results['acc'], 
                          index=[f'alpha={a}' for a in alphas],
                          columns=[f'beta={b}' for b in betas])
    df_nmi = pd.DataFrame(param_results['nmi'], 
                          index=[f'alpha={a}' for a in alphas],
                          columns=[f'beta={b}' for b in betas])
    
    df_acc.to_csv('results/parameter_sensitivity_acc.csv')
    df_nmi.to_csv('results/parameter_sensitivity_nmi.csv')
    
    print("Numerical results saved to CSV files in the 'results' directory")
    
    # Generate final summary report
    print("\n=== Generating final summary report ===")
    generate_summary_report(results, param_results)
    print("Summary report saved to 'results/summary_report.txt'")
    
    # Additional convergence analysis for default parameters (alpha=1.0, beta=1.0)
    print("\n=== Running detailed convergence analysis ===")
    run_convergence_analysis(X, y, n_features=100, alpha=1.0, beta=1.0, max_iter=100)
    print("Detailed convergence analysis completed and saved")


def run_convergence_analysis(X, y, n_features=100, alpha=1.0, beta=1.0, max_iter=100):
    """
    Run a detailed convergence analysis with the specified parameters
    
    Parameters:
    -----------
    X : array-like
        Input data
    y : array-like
        Target labels
    n_features : int
        Number of features to select
    alpha : float
        Weight for l2,1-norm term
    beta : float
        Weight for structure preserving term
    max_iter : int
        Maximum number of iterations
    """
    print(f"Running detailed convergence analysis with alpha={alpha}, beta={beta}, n_features={n_features}")
    
    # Train model with detailed logging
    fs = StructurePreservingFeatureSelection(
        n_features=n_features,
        alpha=alpha,
        beta=beta,
        max_iter=max_iter,
        tol=1e-6,  # Smaller tolerance for more iterations
        verbose=True,
        reg_lambda=1e-4
    )
    
    fs.fit(X)
    
    # Plot and save convergence data
    try:
        df = fs.plot_convergence(save_path='results/detailed_convergence_analysis.png')
        
        # Create additional plots for convergence analysis
        iterations = df['Iteration']
        w_diffs = df['W_Difference']
        obj_values = df['Objective_Value']
        
        # Safety check for extreme values that might cause plotting issues
        obj_values = np.array(obj_values)
        max_reasonable_value = 1e10
        if np.any(obj_values > max_reasonable_value):
            print(f"Warning: Found extremely large objective values (max: {np.max(obj_values)}). Clipping for visualization.")
            obj_values = np.clip(obj_values, None, max_reasonable_value)
        
        # Plot log-scaled W differences
        plt.figure(figsize=(10, 6))
        plt.semilogy(iterations, w_diffs, 'b-o', linewidth=2, alpha=0.7)
        plt.axhline(y=fs.tol, color='r', linestyle='--', alpha=0.5, label=f'Threshold = {fs.tol}')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('W Difference (log scale)', fontsize=12)
        plt.title('Convergence of W Matrix (Log Scale)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig('results/convergence_w_diff_log.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot objective value convergence rate
        if len(obj_values) > 1:
            try:
                # Calculate convergence rate as percentage decrease per iteration
                conv_rates = []
                for i in range(1, len(obj_values)):
                    # Add error handling for potential division by zero or negative numbers
                    if obj_values[i-1] != 0 and obj_values[i-1] > obj_values[i]:
                        rate = (obj_values[i-1] - obj_values[i]) / obj_values[i-1] * 100
                        # Cap extreme rates for better visualization
                        rate = min(rate, 100)
                        conv_rates.append(rate)
                    else:
                        # If current value is higher than previous or previous is zero, set rate to 0
                        conv_rates.append(0)
                
                if len(conv_rates) > 0:
                    plt.figure(figsize=(10, 6))
                    plt.plot(iterations[1:], conv_rates, 'm-o', linewidth=2, alpha=0.7)
                    plt.xlabel('Iteration', fontsize=12)
                    plt.ylabel('Objective Value Decrease (%)', fontsize=12)
                    plt.title('Convergence Rate of Objective Function', fontsize=14)
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    plt.savefig('results/convergence_rate.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # Save convergence rates to CSV
                    import pandas as pd
                    df_rates = pd.DataFrame({
                        'Iteration': iterations[1:],
                        'Convergence_Rate_Pct': conv_rates
                    })
                    df_rates.to_csv('results/convergence_rates.csv', index=False)
            except Exception as e:
                print(f"Warning: Could not generate convergence rate plot: {str(e)}")
        
    except Exception as e:
        print(f"Warning: Error in detailed convergence analysis: {str(e)}")
        print("Will continue with other analyses.")


def generate_summary_report(feature_results, param_results):
    """
    Generate a summary report of all experiments
    
    Parameters:
    -----------
    feature_results : dict
        Results from feature selection experiment
    param_results : dict
        Results from parameter sensitivity analysis
    """
    alphas = param_results['alphas']
    betas = param_results['betas']
    acc_matrix = param_results['acc']
    nmi_matrix = param_results['nmi']
    
    # Find best parameter combinations
    i_max_acc, j_max_acc = np.unravel_index(np.argmax(acc_matrix), acc_matrix.shape)
    i_max_nmi, j_max_nmi = np.unravel_index(np.argmax(nmi_matrix), nmi_matrix.shape)
    
    with open('results/summary_report.txt', 'w') as f:
        f.write("============= Structure Preserving Unsupervised Feature Selection =============\n\n")
        f.write(f"Dataset: MNIST\n")
        f.write(f"Dataset size: {feature_results['n_features'][0]} samples x {feature_results['n_features'][-1]} features\n")
        f.write(f"Number of classes: {len(np.unique(feature_results['acc']))}\n\n")
        
        f.write("--- Feature Selection Results ---\n")
        f.write(f"Baseline (all features): ACC = {feature_results['baseline_acc']:.4f}, NMI = {feature_results['baseline_nmi']:.4f}\n\n")
        
        f.write("Number of Features | ACC      | NMI      \n")
        f.write("----------------------------------\n")
        for i, n_feat in enumerate(feature_results['n_features']):
            f.write(f"{n_feat:<18} | {feature_results['acc'][i]:.6f} | {feature_results['nmi'][i]:.6f}\n")
        
        # Find best number of features
        best_feat_idx = np.argmax(feature_results['acc'])
        best_n_features = feature_results['n_features'][best_feat_idx]
        best_feat_acc = feature_results['acc'][best_feat_idx]
        best_feat_nmi = feature_results['nmi'][best_feat_idx]
        
        f.write(f"\nBest number of features: {best_n_features}\n")
        f.write(f"Best feature selection performance: ACC = {best_feat_acc:.6f}, NMI = {best_feat_nmi:.6f}\n\n")
        
        f.write("--- Parameter Sensitivity Analysis ---\n")
        f.write(f"Best ACC parameters: alpha = {alphas[i_max_acc]}, beta = {betas[j_max_acc]}, ACC = {acc_matrix[i_max_acc, j_max_acc]:.6f}\n")
        f.write(f"Best NMI parameters: alpha = {alphas[i_max_nmi]}, beta = {betas[j_max_nmi]}, NMI = {nmi_matrix[i_max_nmi, j_max_nmi]:.6f}\n\n")
        
        f.write("--- Recommendations ---\n")
        f.write(f"Recommended number of features: {best_n_features}\n")
        f.write(f"Recommended parameters: alpha = {alphas[i_max_acc]}, beta = {betas[j_max_acc]}\n")
        f.write(f"Expected performance: ACC = {acc_matrix[i_max_acc, j_max_acc]:.6f}, NMI = {nmi_matrix[i_max_acc, j_max_acc]:.6f}\n\n")
        
        f.write("============= End of Report =============\n")



if __name__ == "__main__":
    main()