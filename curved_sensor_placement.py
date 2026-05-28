"""
Curved Sensor Placement Simulation
Extension of Kim et al. (IEEE SysCon 2025) to circular arc trajectories
using a 3-D representation space C' = (x0, y0, R).

Author: Hamail Ali, NUST Pakistan
Date: May 2026
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Domain settings
DOMAIN_SIZE = 10.0  # 10km x 10km area
DETECTION_RADIUS = 1.5  # 1.5km sensor range
SIGMA_L = 1.5          # length scale for sensor decay
RHO = 0.95             # max detection probability
NUM_TRAJECTORIES = 300
MU = 3.0               # expected number of target crossings (for Poisson rate)

def generate_curved_trajectories(num_paths: int) -> tuple:
    """Generate circular arc target trajectories passing through the domain.
    Each trajectory is characterized by its center (x0, y0) and radius R.
    """
    np.random.seed(42)
    # Centers are located in a region that produces arcs crossing the [0, 10] x [0, 10] domain
    centers_x = np.random.uniform(-5.0, 15.0, num_paths)
    centers_y = np.random.uniform(-15.0, 5.0, num_paths)
    # Calculate radius to make sure circles pass through the domain center (5, 5) with some variance
    radii = np.sqrt((centers_x - 5.0)**2 + (centers_y - 5.0)**2) + np.random.normal(0, 0.5, num_paths)
    
    # Store parameters in representation space C' = (x0, y0, R)
    representation_points = np.column_stack([centers_x, centers_y, radii])
    return representation_points

def fit_line_to_arc(x0: float, y0: float, R: float) -> tuple:
    """Find the best linear approximation (chord line) of the circular arc
    that passes through the [0, 10] x [0, 10] domain using PCA / Total Least Squares.
    Returns: (alpha, p) such that the line is x cos(alpha) + y sin(alpha) = p.
    """
    # Sample points on the circle
    theta = np.linspace(0, 2 * np.pi, 300)
    px = x0 + R * np.cos(theta)
    py = y0 + R * np.sin(theta)
    
    # Keep points inside the domain
    mask = (px >= 0.0) & (px <= DOMAIN_SIZE) & (py >= 0.0) & (py <= DOMAIN_SIZE)
    pts_x = px[mask]
    pts_y = py[mask]
    
    # Fallback if the arc does not cross or has too few points in the domain
    if len(pts_x) < 3:
        dx = 5.0 - x0
        dy = 5.0 - y0
        dist = np.sqrt(dx**2 + dy**2)
        if dist == 0:
            return 0.0, 5.0
        nx, ny = dx / dist, dy / dist
        p = 5.0 * nx + 5.0 * ny
        alpha = np.arctan2(ny, nx)
        return alpha, p
        
    # Find the principal component of the points to fit the line
    mx, my = np.mean(pts_x), np.mean(pts_y)
    dx = pts_x - mx
    dy = pts_y - my
    
    cov = np.cov(dx, dy)
    if cov.ndim == 0 or np.allclose(cov, 0):
        dx_c = 5.0 - x0
        dy_c = 5.0 - y0
        dist = np.sqrt(dx_c**2 + dy_c**2)
        if dist == 0:
            return 0.0, 5.0
        nx, ny = dx_c / dist, dy_c / dist
        p = 5.0 * nx + 5.0 * ny
        alpha = np.arctan2(ny, nx)
        return alpha, p
        
    eigvals, eigvecs = np.linalg.eigh(cov)
    # The normal direction to the line is the eigenvector of the smallest eigenvalue
    normal = eigvecs[:, 0]
    cos_a, sin_a = normal[0], normal[1]
    
    p = mx * cos_a + my * sin_a
    alpha = np.arctan2(sin_a, cos_a)
    return alpha, p

def linear_assumed_distance(sensors: np.ndarray, representation_points: np.ndarray) -> np.ndarray:
    """Calculate the shortest distance from sensors to the linear approximations of the arcs.
    Implements the original paper's linear model parameterization.
    """
    num_sensors = len(sensors)
    num_paths = len(representation_points)
    dists = np.zeros((num_paths, num_sensors))
    
    for j, (x0, y0, R) in enumerate(representation_points):
        alpha, p = fit_line_to_arc(x0, y0, R)
        cos_a = np.cos(alpha)
        sin_a = np.sin(alpha)
        for i, (ax, ay) in enumerate(sensors):
            dists[j, i] = np.abs(ax * cos_a + ay * sin_a - p)
            
    return dists

def curved_distance(sensors: np.ndarray, representation_points: np.ndarray) -> np.ndarray:
    """Calculate the exact shortest distance from a sensor to the circular trajectory."""
    if len(sensors) == 0:
        return np.zeros((len(representation_points), 0))
    diff_x = representation_points[:, 0, np.newaxis] - sensors[np.newaxis, :, 0]
    diff_y = representation_points[:, 1, np.newaxis] - sensors[np.newaxis, :, 1]
    dist_to_center = np.sqrt(diff_x**2 + diff_y**2)
    dists = np.abs(dist_to_center - representation_points[:, 2, np.newaxis])
    return dists

def estimate_kde_weights(train_points: np.ndarray, eval_points: np.ndarray) -> np.ndarray:
    """Compute 3-D Kernel Density Estimation weights for the evaluation trajectories
    to represent the target intensity function lambda(l) parametric fit.
    """
    N_train = len(train_points)
    N_eval = len(eval_points)
    
    # Calculate standard deviation for each dimension
    std_x = np.std(train_points[:, 0])
    std_y = np.std(train_points[:, 1])
    std_R = np.std(train_points[:, 2])
    
    # Scott's Rule for bandwidth Selection
    bx = std_x * (N_train ** (-1.0 / 7.0))
    by = std_y * (N_train ** (-1.0 / 7.0))
    bR = std_R * (N_train ** (-1.0 / 7.0))
    
    # Avoid division by zero
    bx = max(bx, 0.1)
    by = max(by, 0.1)
    bR = max(bR, 0.1)
    
    # Compute density for each evaluation point using 3-D Gaussian kernel
    densities = np.zeros(N_eval)
    for i, c in enumerate(eval_points):
        kx = np.exp(-0.5 * ((c[0] - train_points[:, 0]) / bx) ** 2) / (bx * np.sqrt(2 * np.pi))
        ky = np.exp(-0.5 * ((c[1] - train_points[:, 1]) / by) ** 2) / (by * np.sqrt(2 * np.pi))
        kR = np.exp(-0.5 * ((c[2] - train_points[:, 2]) / bR) ** 2) / (bR * np.sqrt(2 * np.pi))
        densities[i] = np.mean(kx * ky * kR)
        
    # Scale densities to match the expected number of target crossings MU
    mean_density = np.mean(densities)
    if mean_density > 0:
        weights = (densities / mean_density) * (MU / N_eval)
    else:
        weights = np.ones(N_eval) * (MU / N_eval)
        
    return weights

def expected_void_probability_approx(sensors: np.ndarray, representation_points: np.ndarray, weights: np.ndarray, mode: str = 'curved') -> float:
    """Compute the void probability approximation using the thinned intensity.
    Returns: P_void(S) = exp( - sum pi_C * w_i )
    """
    if len(sensors) == 0:
        return np.exp(-MU) # No sensors deployed
    
    if mode == 'curved':
        dists = curved_distance(sensors, representation_points)
    else:
        dists = linear_assumed_distance(sensors, representation_points)
        
    # Sensor detection probability decaying exponentially
    gamma = RHO * np.exp(-(dists**2) / SIGMA_L)
    # Probability of escaping the sensor network
    pi_C = np.prod(1.0 - gamma, axis=1)  # (N_eval,)
    
    # Expected void probability of thinned Poisson process
    return np.exp(-np.sum(pi_C * weights))

def expected_detection_probability(sensors: np.ndarray, representation_points: np.ndarray, weights: np.ndarray, mode: str = 'curved') -> float:
    """Compute the expected detection probability of a target:
    P_det(S) = 1.0 - (expected undetected targets / MU)
             = 1.0 + ln(P_void(S)) / MU
    """
    if len(sensors) == 0:
        return 0.0
    p_void = expected_void_probability_approx(sensors, representation_points, weights, mode)
    return 1.0 + np.log(p_void) / MU

def greedy_placement(num_sensors: int, representation_points: np.ndarray, weights: np.ndarray, mode: str = 'curved', grid_res: int = 20) -> np.ndarray:
    """Greedy sensor placement in the domain [0, 10] x [0, 10]."""
    lin = np.linspace(0.5, DOMAIN_SIZE - 0.5, grid_res)
    xv, yv = np.meshgrid(lin, lin)
    candidates = np.column_stack([xv.ravel(), yv.ravel()])
    
    selected = []
    for _ in range(num_sensors):
        best_val = -1.0
        best_cand = None
        for cand in candidates:
            if any(np.allclose(cand, s) for s in selected):
                continue
            trial = np.vstack([selected, cand]) if selected else np.array([cand])
            # Maximize the expected detection probability
            val = expected_detection_probability(trial, representation_points, weights, mode)
            if val > best_val:
                best_val = val
                best_cand = cand
        selected.append(best_cand)
    return np.array(selected)

def random_placement(num_sensors: int) -> np.ndarray:
    """Generate random sensor placements."""
    np.random.seed(1337)
    return np.random.uniform(0.5, DOMAIN_SIZE - 0.5, (num_sensors, 2))

def run_simulation():
    os.makedirs('results', exist_ok=True)
    all_points = generate_curved_trajectories(NUM_TRAJECTORIES)
    
    # Split trajectories into 150 historical training paths and 150 test evaluation paths
    train_points = all_points[:150]
    eval_points = all_points[150:]
    
    # Calculate 3-D KDE weights for evaluation
    weights = estimate_kde_weights(train_points, eval_points)
    
    sensor_counts = [1, 2, 3, 4, 5]
    results_rand = []
    results_linear = []
    results_curved = []
    
    print("Evaluating sensor placements...")
    for K in sensor_counts:
        # 1. Random Placement
        sensors_rand = random_placement(K)
        val_rand = expected_detection_probability(sensors_rand, eval_points, weights, 'curved')
        results_rand.append(val_rand)
        
        # 2. Linear-Assumed Greedy (Original paper baseline)
        sensors_linear = greedy_placement(K, eval_points, weights, 'linear')
        val_linear = expected_detection_probability(sensors_linear, eval_points, weights, 'curved')
        results_linear.append(val_linear)
        
        # 3. Curved-Aware Greedy (Our extension)
        sensors_curved = greedy_placement(K, eval_points, weights, 'curved')
        val_curved = expected_detection_probability(sensors_curved, eval_points, weights, 'curved')
        results_curved.append(val_curved)
        
        print(f"K={K} | Random: {val_rand:.4f} | Linear-Assumed: {val_linear:.4f} | Curved-Aware: {val_curved:.4f}")
        
    # Write to a text file for verification
    with open('results/simulation_results.txt', 'w') as f:
        f.write("K,Random,Linear-Assumed,Curved-Aware\n")
        for idx, K in enumerate(sensor_counts):
            f.write(f"{K},{results_rand[idx]:.6f},{results_linear[idx]:.6f},{results_curved[idx]:.6f}\n")
    print("Numerical results saved to results/simulation_results.txt")
        
    # Plot results
    plt.figure(figsize=(7, 5))
    plt.plot(sensor_counts, results_rand, 'r-o', label='Random Placement', linewidth=1.5)
    plt.plot(sensor_counts, results_linear, 'g-s', label='Linear-Assumed Greedy (Original Paper)', linewidth=1.5)
    plt.plot(sensor_counts, results_curved, 'b-^', label='Curved-Aware Greedy (Our Proposed Extension)', linewidth=1.7)
    plt.xlabel('Number of Sensors (K)', fontsize=11)
    plt.ylabel('Expected Detection Probability P_det(S)', fontsize=11)
    plt.title('Sensor Placement on Curved (Circular Arc) Trajectories', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', frameon=True, shadow=True)
    plt.ylim(0, 1.0)
    plt.savefig('results/curved_coverage_vs_sensors.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # Save a visualization of placements for K=4
    plt.figure(figsize=(10, 5))
    
    # Linear Assumed (places sensors under straight-line assumption)
    plt.subplot(1, 2, 1)
    s_lin = greedy_placement(4, eval_points, weights, 'linear')
    plt.scatter(s_lin[:, 0], s_lin[:, 1], c='green', s=100, label='Sensors', zorder=3)
    for x, y in s_lin:
        circle = plt.Circle((x, y), DETECTION_RADIUS, color='green', alpha=0.15, zorder=2)
        plt.gca().add_patch(circle)
    # Plot a few curved trajectories (circles)
    for j in range(10):
        x0, y0, R = eval_points[j]
        theta = np.linspace(0, 2*np.pi, 100)
        cx = x0 + R * np.cos(theta)
        cy = y0 + R * np.sin(theta)
        plt.plot(cx, cy, 'k--', alpha=0.2)
    plt.xlim(0, DOMAIN_SIZE)
    plt.ylim(0, DOMAIN_SIZE)
    plt.title('Linear-Assumed Placement (K=4)')
    plt.xlabel('x (km)')
    plt.ylabel('y (km)')
    plt.gca().set_aspect('equal')
    plt.grid(True, linestyle=':', alpha=0.5)
    
    # Curved Aware (our model)
    plt.subplot(1, 2, 2)
    s_curv = greedy_placement(4, eval_points, weights, 'curved')
    plt.scatter(s_curv[:, 0], s_curv[:, 1], c='blue', s=100, label='Sensors', zorder=3)
    for x, y in s_curv:
        circle = plt.Circle((x, y), DETECTION_RADIUS, color='blue', alpha=0.15, zorder=2)
        plt.gca().add_patch(circle)
    # Plot the same trajectories
    for j in range(10):
        x0, y0, R = eval_points[j]
        theta = np.linspace(0, 2*np.pi, 100)
        cx = x0 + R * np.cos(theta)
        cy = y0 + R * np.sin(theta)
        plt.plot(cx, cy, 'k--', alpha=0.2)
    plt.xlim(0, DOMAIN_SIZE)
    plt.ylim(0, DOMAIN_SIZE)
    plt.title('Curved-Aware Placement (K=4)')
    plt.xlabel('x (km)')
    plt.gca().set_aspect('equal')
    plt.grid(True, linestyle=':', alpha=0.5)
    
    plt.savefig('results/curved_placement_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    run_simulation()