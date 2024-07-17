from filterpy.kalman import KalmanFilter
import numpy as np

def createKF(x, y, s, r):
    """Initializes a Kalman Filter for a new track based on initial (x, y) position, scale, and aspect ratio. This is alligned with SORT method"""
    kf = KalmanFilter(dim_x=6, dim_z=4)
    
    # State transition matrix
    dt = 1  # time step
    kf.F = np.array([[1, 0, dt, 0, 0, 0], 
                     [0, 1, 0, dt, 0, 0], 
                     [0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1]])
    
    # Measurement function
    kf.H = np.array([[1, 0, 0, 0, 0, 0], 
                     [0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1]])
    
    # Initial state estimate
    kf.x = np.array([x, y, 0, 0, s, r]).T
    
    # Covariance matrix
    kf.P *= 1000
    
    # Measurement noise
    kf.R = np.array([[5, 0, 0, 0], 
                     [0, 5, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    
    # Process noise
    kf.Q = np.array([[dt**4/4, 0, dt**3/2, 0, 0, 0],
                     [0, dt**4/4, 0, dt**3/2, 0, 0],
                     [dt**3/2, 0, dt**2, 0, 0, 0],
                     [0, dt**3/2, 0, dt**2, 0, 0],
                     [0, 0, 0, 0, 0.01, 0],
                     [0, 0, 0, 0, 0, 0.01]]) * 0.03
    
    return kf


def KFTrustworthy(track, variance_thresholds):
    p_diag = np.diag(track.kf.P)  # Extract the diagonal (variances) from the covariance matrix
    for variance, threshold in zip(p_diag, variance_thresholds):
        if variance > threshold:
            return False  # Prediction is not trustworthy if any variance exceeds its threshold
    return True  # All variances are within their thresholds