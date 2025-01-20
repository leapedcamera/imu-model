import yaml
import numpy as np


class DarkImu():
    """
    A simple IMU Model, errors included:
    - Accel/Gyro Bias
    - Accel/Gyro Bias Instability
    - Accel/Gyro Scale Factor
    - Accel/Gyro G-Sensitivity
    - Accel/Gyro Noise (VRW/ARW)
    
    Methods
    -------

    fogm(self, tau, sigma, prev, dt)
        First Order Gauss-Markov model used to simulate bias instability
        Taken from  Nassar (2003), Modeling Inertial Sensor Errors Using Autoregressive (AR) Models
        x_dot = -1/tau * x + w
       
        Parameters
        tau: the correlation time constant
        sigma: the 1sigma of the white noise driving the random process
        prev: the previous instability/drift value

     getImuOutput(self, w, sf, dt)
        Simulates the response of the IMU to input
        
        Parameters
        w: The average angular rate over the dt
        sf: The average specific force over the dt, taken in the body frame at the midpoint of the sample period
        dt: The sample period, the reciprocal of the frequency of the device


    """
    def __init__(self, paramFile):
        with open(paramFile, 'r') as file:       
            data = yaml.safe_load(file)

        tempVal = data["accelBiasSigma"] # gs
        self.ab = tempVal * np.random.randn(3)
                          
        tempVal = data["accelSfSigma"] # PPM
        self.asf = np.identity(3) + 1e-6 * np.diag(tempVal * np.random.randn(3))
        
        tempVal = data["accelG2Sigma"] # PPM / g
        self.ag2 = np.diag(tempVal * np.random.randn(3) )
        
        self.vrw = data["vrw"] # g/sqrt(Hz)

        self.atau = data["accelCorrelationTime"] # seconds
        self.abdSigma = data["accelBiasDriftSigma"] # gs
        self.prevAccelBiasDrift = np.array([0,0,0])

        tempVal = data["gyroBiasSigma"] # rad/s
        self.gb = tempVal * np.random.randn(3) 
                          
        tempVal = data["gyroSfSigma"] # PPM
        self.gsf = np.identity(3) + 1e-6 * np.diag(tempVal * np.random.randn(3))
        
        tempVal = data["gyroG2Sigma"] # PPM/g
        self.gg2 = np.diag(tempVal * np.random.randn(3) )
        
        self.arw = data["arw"] # rad/s/sqrt(Hz)

        self.gtau = data["gyroCorrelationTime"] #s
        self.gbdSigma  = data["gyroBiasDriftSigma"] # rad/s
        self.prevGyroBiasDrift = np.array([0,0,0])

    
    def getImuOutput(self, w, sf, dt):

        # Calculate the instability first,
        currAccelDrift = self.fogm(self.atau, self.abdSigma, self.prevAccelBiasDrift, dt)
        
        # Combine all error terms into sensor reading
        fs = self.asf @ sf + self.ag2 @ sf + self.ab + currAccelDrift + \
            self.vrw / np.sqrt(dt) * np.random.randn(3)
        
        # Convert to traditional data output of IMU
        deltaVel = fs * dt * 9.81

        # Calculate the instability first,
        currGyroDrift = self.fogm(self.gtau, self.gbdSigma, self.prevGyroBiasDrift, dt)      

        # Combine all error terms into sensor reading
        wS = self.gsf @ w + self.gg2 @ sf + self.gb + currGyroDrift + \
            self.arw / np.sqrt(dt) * np.random.randn(3)
        deltaTheta = wS * dt

        self.prevGyroBiasDrift = currGyroDrift
        self.prevAccelBiasDrift = currAccelDrift
        return np.hstack((deltaVel,deltaTheta))
    
    def fogm(self, tau, sigma, prev, dt):
        beta = 1 / tau
        return  (1 - beta * dt)  * prev + np.sqrt(2 * beta * sigma * sigma) * np.random.randn(3) * dt
        