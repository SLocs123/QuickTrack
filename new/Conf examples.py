#--------------------------------------------------------------------------------------#
# example util file

# util.py

# Regular confidence calculation functions
def conf_a():
    # Simulated confidence calculation
    return 0.75  # Example return value

def conf_b():
    return 0.65

def conf_c():
    return 0.80

# Vital confidence calculation functions
def confVital_a():
    # Simulated vital confidence calculation
    return 1  # Example return value

def confVital_b():
    return 0.9

# Additional conf_ and confVital_ functions can be added as needed
#--------------------------------------------------------------------------------------#
# weighted confidence, no vital functions

import util
import inspect

class Tracks:
    def __init__(self, weights):
        # Get only functions from util.py that start with 'conf_'
        # Store them in the self.confMetrics attribute
        self.confMetrics = [func for func in dir(util) if callable(getattr(util, func)) and func.startswith("conf_")]

        if len(weights) != len(self.confMetrics):
            raise ValueError("The number of weights must match the number of confidence functions.")
        
        self.weights = weights

    def calculate_weighted_confidence(self):
        # Calculate weighted average of confidences
        total_confidence = 0
        total_weight = sum(self.weights)
        for func, weight in zip(self.confMetrics, self.weights):
            function = getattr(util, func)
            total_confidence += function() * weight

        # Normalize by total weight
        weighted_confidence = total_confidence / total_weight
        return weighted_confidence

#--------------------------------------------------------------------------------------#
# weighted conf, 0 if vital not met

import util
import inspect

class Tracks:
    def __init__(self, weights):
        # Get only functions from util.py that start with 'conf_'
        self.confMetrics = [func for func in dir(util) if callable(getattr(util, func)) and func.startswith("conf_")]
        # Identify vital functions that start with 'confVital_'
        self.vitalFunctions = [func for func in dir(util) if callable(getattr(util, func)) and func.startswith("confVital_")]

        if len(weights) != len(self.confMetrics):
            raise ValueError("The number of weights must match the number of confidence functions.")
        
        self.weights = weights

    def calculate_weighted_confidence(self):
        # Check if any vital function does not meet the required metric (i.e., does not return 1)
        for vitalFunc in self.vitalFunctions:
            vitalFunction = getattr(util, vitalFunc)
            if vitalFunction() != 1:
                return 0  # Set overall confidence to 0 if any vital condition is not met

        # Calculate weighted average of confidences for all functions
        total_confidence = 0
        total_weight = sum(self.weights)
        for func, weight in zip(self.confMetrics, self.weights):
            function = getattr(util, func)
            total_confidence += function() * weight

        # Normalize by total weight
        weighted_confidence = total_confidence / total_weight
        return weighted_confidence
#--------------------------------------------------------------------------------------#
# weighted conf, scalled down if vital not met

import util
import inspect

class Tracks:
    def __init__(self, weights, scale_factor):
        # Get only functions from util.py that start with 'conf_'
        self.confMetrics = [func for func in dir(util) if callable(getattr(util, func)) and func.startswith("conf_")]
        # Identify vital functions that start with 'confVital_'
        self.vitalFunctions = [func for func in dir(util) if callable(getattr(util, func)) and func.startswith("confVital_")]

        if len(weights) != len(self.confMetrics):
            raise ValueError("The number of weights must match the number of confidence functions.")
        
        self.weights = weights
        self.scale_factor = scale_factor

    def calculate_weighted_confidence(self):
        # Initialize a flag to track if all vital conditions are met
        all_vital_conditions_met = True

        # Check vital functions
        for vitalFunc in self.vitalFunctions:
            vitalFunction = getattr(util, vitalFunc)
            if vitalFunction() != 1:
                all_vital_conditions_met = False
                break

        # Calculate weighted average of confidences for all functions
        total_confidence = 0
        total_weight = sum(self.weights)
        for func, weight in zip(self.confMetrics, self.weights):
            function = getattr(util, func)
            total_confidence += function() * weight

        # Normalize by total weight
        weighted_confidence = total_confidence / total_weight

        # Scale down the confidence if not all vital conditions are met
        if not all_vital_conditions_met:
            weighted_confidence *= self.scale_factor

        return weighted_confidence
#--------------------------------------------------------------------------------------#
# weighted conf, weighted vitals, scalled down

import util
import inspect

class Tracks:
    def __init__(self, normalWeights, vitalWeights):
        # Get all functions from util.py that start with 'conf_'
        self.confMetrics = [func for func in dir(util) if callable(getattr(util, func)) and func.startswith("conf_")]
        # Identify vital functions that start with 'confVital_'
        self.vitalFunctions = [func for func in dir(util) if callable(getattr(util, func)) and func.startswith("confVital_")]

        if len(normalWeights) != len(self.confMetrics):
            raise ValueError("The number of normal weights must match the number of confidence functions.")
        if len(vitalWeights) != len(self.vitalFunctions):
            raise ValueError("The number of vital weights must match the number of vital confidence functions.")
        
        self.normalWeights = normalWeights
        self.vitalWeights = vitalWeights

    def calculate_weighted_confidence(self):
        # Calculate weighted average of confidences for all functions
        total_confidence = 0
        total_weight = sum(self.normalWeights)
        for func, weight in zip(self.confMetrics, self.normalWeights):
            function = getattr(util, func)
            total_confidence += function() * weight

        # Normalize by total weight
        weighted_confidence = total_confidence / total_weight

        # Adjust confidence based on vital functions and their weights
        vital_total = sum([getattr(util, func)() * weight for func, weight in zip(self.vitalFunctions, self.vitalWeights)])
        vital_weight = sum(self.vitalWeights)
        vital_adjustment = vital_total / vital_weight if vital_weight > 0 else 0

        # Scale the final confidence based on the vital adjustment
        final_confidence = weighted_confidence * vital_adjustment
        return final_confidence

#--------------------------------------------------------------------------------------#
