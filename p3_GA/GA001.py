"""
Genetic Algorithm XGBoost Inverse Optimization System
Function: Use Genetic Algorithm to find optimal input feature values for XGBoost model output
Support: Load feature information from model info JSON files
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os
import warnings
from datetime import datetime
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import sys
import atexit
import time
import random

# Genetic Algorithm library
from deap import base, creator, tools, algorithms  # Requires installation: pip install deap

# Suppress warnings
warnings.filterwarnings('ignore')


# ====================== Configuration Class ======================
class Config:
    """Configuration Settings"""
    # Project root directory
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    # Output directory
    GA_RESULTS_DIR = os.path.join(PROJECT_ROOT, "ga_optimization_results")

    # GA default parameters
    GA_DEFAULT_PARAMS = {
        'population_size': 50,      # Population size
        'generations': 100,         # Maximum generations
        'crossover_prob': 0.8,      # Crossover probability
        'mutation_prob': 0.2,       # Mutation probability
        'tournament_size': 3,       # Tournament selection size
        'eta': 10.0,                # Parameter for crossover (SBX) and mutation
        'elitism_ratio': 0.1,       # Ratio of elites to keep
        'min_value': 0.0,           # Minimum value for initialization
        'max_value': 1.0,           # Maximum value for initialization
        'hof_size': 5               # Hall of fame size
    }

    # Ensure directories exist
    @staticmethod
    def ensure_directories():
        """Ensure all necessary directories exist"""
        directories = [
            Config.GA_RESULTS_DIR
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"Directory ensured: {directory}")

        return True


# ====================== XGBoost Model Loader ======================
class XGBoostModelLoader:
    """XGBoost Model Loader"""

    def __init__(self, log_callback=None):
        self.model = None
        self.model_info = None
        self.selected_features = []
        self.target_variable = "Target"
        self.feature_ranges = {}
        self.log_callback = log_callback
        self.model_params = {}
        # Add data scalers for reverse transformation
        self.feature_scaler = None
        self.target_scaler = None
        self.original_feature_stats = {}  # Store original feature statistics
        self.original_feature_names = []  # Store original feature names
        self.user_modified_ranges = {}  # Track user-modified ranges
        self.json_feature_info = {}  # Feature information loaded from JSON

    def log(self, message):
        """Log message"""
        if self.log_callback:
            self.log_callback(message)

    def _ensure_dict(self, obj):
        """Ensure object is a dictionary type"""
        if isinstance(obj, dict):
            return obj
        elif hasattr(obj, 'items'):
            # If it's dict_items or other view, convert to dict
            return dict(obj)
        else:
            # If it's a list or other type, try conversion
            try:
                return dict(obj)
            except:
                self.log(f"Warning: Cannot convert {type(obj)} to dictionary")
                return {}

    def _format_float(self, value):
        """Format float value to max 3 decimal places, removing trailing zeros"""
        try:
            # Convert to string with maximum 3 decimal places
            formatted = f"{float(value):.3f}"
            # Remove trailing zeros and decimal point if not needed
            if '.' in formatted:
                formatted = formatted.rstrip('0').rstrip('.')
            return formatted
        except:
            return str(value)

    def load_model(self, model_path):
        """Load XGBoost model"""
        try:
            self.log(f"Loading model: {model_path}")

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Load model
            self.model = joblib.load(model_path)
            self.log(f"✓ Model loaded successfully")

            # Get model parameters
            self.model_params = self.model.get_params()

            # Try to load model info file
            info_path = self._find_model_info_file(model_path)
            if info_path and os.path.exists(info_path):
                with open(info_path, 'r', encoding='utf-8') as f:
                    self.model_info = json.load(f)
                self.log(f"✓ Model info loaded successfully")

                # Extract key information
                self.target_variable = self.model_info.get('target_variable', 'Target')
                self.selected_features = self.model_info.get('selected_features', [])

                # Get original feature names
                self.original_feature_names = self.model_info.get('original_feature_names',
                                                                  self.model_info.get('feature_names',
                                                                                      self.selected_features.copy()))

                # Try to load scalers
                scaler_path = self._find_scaler_file(model_path)
                if scaler_path and os.path.exists(scaler_path):
                    try:
                        scalers = joblib.load(scaler_path)
                        if isinstance(scalers, dict):
                            self.feature_scaler = scalers.get('feature_scaler')
                            self.target_scaler = scalers.get('target_scaler')
                        else:
                            self.feature_scaler = scalers
                        self.log("✓ Scalers loaded successfully")
                    except Exception as e:
                        self.log(f"Scaler loading failed: {e}")

                # Get original feature statistics
                self._extract_feature_stats_from_model_info()

            # If still no feature info, try to extract from model itself
            if not self.selected_features:
                self._extract_features_from_model()

            # Ensure we have original feature names
            if not self.original_feature_names:
                self.original_feature_names = self.selected_features.copy()

            # Initialize default feature ranges to 0~1
            self._initialize_default_feature_ranges()

            return True, "Model loaded successfully"

        except Exception as e:
            return False, f"Model loading failed: {str(e)}"

    def _extract_feature_stats_from_model_info(self):
        """Extract feature statistics from model info"""
        if not self.model_info:
            return

        # Try to extract feature statistics from different locations
        if 'scaling_parameters' in self.model_info:
            scaling_params = self.model_info['scaling_parameters']

            # Check if means and stds exist
            if 'means' in scaling_params and 'stds' in scaling_params:
                means = scaling_params['means']
                stds = scaling_params['stds']

                # Ensure means and stds are dictionary types
                means_dict = self._ensure_dict(means)
                stds_dict = self._ensure_dict(stds)

                if isinstance(means_dict, dict) and isinstance(stds_dict, dict):
                    # Iterate through all features, collect statistics
                    for feature_name in means_dict.keys():
                        if feature_name in stds_dict:
                            stats = {
                                'mean': float(means_dict[feature_name]),
                                'std': float(stds_dict[feature_name])
                            }
                            self.original_feature_stats[feature_name] = stats

                    self.log(f"Extracted statistics for {len(means_dict)} features from scaling_parameters")
                else:
                    self.log("Warning: means/stds in scaling_parameters are not dictionary format")

        # If we have original feature names but no statistics, at least create empty records
        if self.original_feature_names and not self.original_feature_stats:
            for feature_name in self.original_feature_names:
                self.original_feature_stats[feature_name] = {'mean': 0.0, 'std': 1.0}
            self.log(f"Created default statistics for {len(self.original_feature_names)} features")

    def load_feature_info_json(self, json_path):
        """Load feature information JSON file"""
        try:
            self.log(f"Loading feature information from: {json_path}")

            if not os.path.exists(json_path):
                raise FileNotFoundError(f"JSON file not found: {json_path}")

            with open(json_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)

            # Ensure loaded data is dictionary type
            if not isinstance(loaded_data, dict):
                self.log(f"Warning: JSON file content is not dictionary format: {type(loaded_data)}")
                # Try to convert or process
                loaded_data = dict(loaded_data) if hasattr(loaded_data, 'items') else {}

            self.json_feature_info = loaded_data
            self.log(f"✓ JSON file loaded successfully")

            # Parse feature information
            self._parse_json_feature_info()

            return True, f"Feature information loaded successfully: {len(self.original_feature_names)} features"

        except Exception as e:
            return False, f"Feature information JSON loading failed: {str(e)}"

    def _parse_json_feature_info(self):
        """Parse JSON feature information"""
        if not self.json_feature_info:
            return

        # Clear existing feature information
        self.original_feature_names = []
        self.original_feature_stats = {}

        # First check if it's a complete model info file
        if 'original_feature_names' in self.json_feature_info:
            # This is a model info file, use original_feature_names directly
            original_names = self.json_feature_info['original_feature_names']
            # Ensure it's a list type
            if isinstance(original_names, list):
                self.original_feature_names = original_names
                self.log(f"Obtained {len(self.original_feature_names)} feature names from original_feature_names")
            else:
                self.log(f"Warning: original_feature_names is not list type: {type(original_names)}")

            # Extract feature statistics
            self._extract_stats_from_model_info_json()

        # Handle JSON format containing scaling_parameters
        elif 'scaling_parameters' in self.json_feature_info:
            scaling_params = self.json_feature_info['scaling_parameters']

            if 'means' in scaling_params and 'stds' in scaling_params:
                means = scaling_params['means']
                stds = scaling_params['stds']

                # Ensure means and stds are dictionary types
                means_dict = self._ensure_dict(means)
                stds_dict = self._ensure_dict(stds)

                if isinstance(means_dict, dict) and isinstance(stds_dict, dict):
                    # Ensure means and stds have the same keys
                    means_keys = set(means_dict.keys())
                    stds_keys = set(stds_dict.keys())

                    if means_keys == stds_keys:
                        for feature_name in means_dict.keys():
                            # Update original feature names
                            if feature_name not in self.original_feature_names:
                                self.original_feature_names.append(feature_name)

                            # Update feature statistics
                            stats = {
                                'mean': float(means_dict[feature_name]),
                                'std': float(stds_dict[feature_name])
                            }
                            self.original_feature_stats[feature_name] = stats

                        self.log(f"Parsed {len(means_dict)} features from scaling_parameters")
                    else:
                        self.log(
                            f"Warning: means and stds keys don't match: {means_keys - stds_keys} vs {stds_keys - means_keys}")
                else:
                    self.log("Warning: means and stds are not dictionary format")

        # Handle JSON format containing means and stds (root level)
        elif 'means' in self.json_feature_info and 'stds' in self.json_feature_info:
            means = self.json_feature_info['means']
            stds = self.json_feature_info['stds']

            # Ensure means and stds are dictionary types
            means_dict = self._ensure_dict(means)
            stds_dict = self._ensure_dict(stds)

            if isinstance(means_dict, dict) and isinstance(stds_dict, dict):
                # Ensure means and stds have the same keys
                means_keys = set(means_dict.keys())
                stds_keys = set(stds_dict.keys())

                if means_keys == stds_keys:
                    for feature_name in means_dict.keys():
                        # Update original feature names
                        if feature_name not in self.original_feature_names:
                            self.original_feature_names.append(feature_name)

                        # Update feature statistics
                        stats = {
                            'mean': float(means_dict[feature_name]),
                            'std': float(stds_dict[feature_name])
                        }
                        self.original_feature_stats[feature_name] = stats

                    self.log(f"Parsed {len(means_dict)} features from means/stds")
                else:
                    self.log(
                        f"Warning: means and stds keys don't match: {means_keys - stds_keys} vs {stds_keys - means_keys}")
            else:
                self.log("Warning: means and stds are not dictionary format")

        # If still no feature names, try to get from other locations
        if not self.original_feature_names and 'selected_features' in self.json_feature_info:
            selected = self.json_feature_info['selected_features']
            if isinstance(selected, list):
                self.original_feature_names = selected
                self.log(f"Obtained {len(self.original_feature_names)} feature names from selected_features")

    def _extract_stats_from_model_info_json(self):
        """Extract statistics from model info JSON"""
        if not self.json_feature_info:
            return

        # Try to extract statistics from scaling_parameters
        if 'scaling_parameters' in self.json_feature_info:
            scaling_params = self.json_feature_info['scaling_parameters']

            if 'means' in scaling_params and 'stds' in scaling_params:
                means = scaling_params['means']
                stds = scaling_params['stds']

                # Ensure means and stds are dictionary types
                means_dict = self._ensure_dict(means)
                stds_dict = self._ensure_dict(stds)

                if isinstance(means_dict, dict) and isinstance(stds_dict, dict):
                    for feature_name in self.original_feature_names:
                        if feature_name in means_dict and feature_name in stds_dict:
                            stats = {
                                'mean': float(means_dict[feature_name]),
                                'std': float(stds_dict[feature_name])
                            }
                            self.original_feature_stats[feature_name] = stats

                    self.log(
                        f"Extracted statistics for {len(self.original_feature_stats)} features from model info JSON")
                else:
                    self.log("Warning: means/stds in scaling_parameters are not dictionary format")

        # If still no statistics, create defaults
        if not self.original_feature_stats and self.original_feature_names:
            for feature_name in self.original_feature_names:
                self.original_feature_stats[feature_name] = {'mean': 0.0, 'std': 1.0}
            self.log(f"Created default statistics for {len(self.original_feature_names)} features")

    def _find_model_info_file(self, model_path):
        """Find model info file"""
        possible_paths = [
            model_path.replace('.pkl', '_info.json'),
            model_path.replace('.pkl', '.json'),
            os.path.join(os.path.dirname(model_path), 'model_info.json'),
            os.path.join(os.path.dirname(model_path), 'training_summary.json'),
            os.path.join(os.path.dirname(model_path), 'model_metadata.json'),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None

    def _find_scaler_file(self, model_path):
        """Find scaler file"""
        possible_paths = [
            model_path.replace('.pkl', '_scaler.pkl'),
            model_path.replace('.pkl', '_scalers.pkl'),
            os.path.join(os.path.dirname(model_path), 'scaler.pkl'),
            os.path.join(os.path.dirname(model_path), 'scalers.pkl'),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None

    def _extract_features_from_model(self):
        """Extract feature information from model"""
        try:
            if hasattr(self.model, 'feature_importances_'):
                num_features = len(self.model.feature_importances_)
                self.selected_features = [f"Feature_{i + 1}" for i in range(num_features)]
                self.log(f"Inferred {num_features} features from feature importance")
            elif hasattr(self.model, 'get_booster'):
                booster = self.model.get_booster()
                if hasattr(booster, 'feature_names') and booster.feature_names:
                    self.selected_features = list(booster.feature_names)
                    self.log(f"Obtained {len(self.selected_features)} features from booster")
            else:
                # If no feature info, use default count
                self.selected_features = ["Feature_1", "Feature_2", "Feature_3", "Feature_4", "Feature_5"]
                self.log(f"Using default 5 features")
        except Exception as e:
            self.log(f"Feature extraction error: {e}")
            self.selected_features = ["Feature_1", "Feature_2", "Feature_3"]

    def _initialize_default_feature_ranges(self):
        """Initialize default feature ranges to 0~1"""
        for i, feature in enumerate(self.selected_features):
            # If we already have JSON feature names, use JSON feature names
            if i < len(self.original_feature_names):
                feature_name = self.original_feature_names[i]
            else:
                feature_name = feature

            # Always use 0~1 as default range
            self.feature_ranges[feature] = [0.0, 1.0]
            self.log(f"Set default range for {feature_name}: [0.0, 1.0]")

        self.log(f"Initialized {len(self.feature_ranges)} feature ranges to [0,1]")

    def set_feature_ranges(self, feature_ranges):
        """Set feature ranges (allow user modification)"""
        self.user_modified_ranges = feature_ranges.copy()

        # Apply user modifications
        for feature, range_val in feature_ranges.items():
            if isinstance(range_val, (list, tuple)) and len(range_val) >= 2:
                min_val = float(range_val[0])
                max_val = float(range_val[1])

                if min_val >= max_val:
                    self.log(f"Adjusting invalid range for {feature}")
                    center = (min_val + max_val) / 2
                    min_val = center - 1.0
                    max_val = center + 1.0

                self.feature_ranges[feature] = [min_val, max_val]
            else:
                self.log(f"Invalid range for {feature}, using original range")

        # Fill in missing features
        for i, feature in enumerate(self.selected_features):
            if feature not in self.feature_ranges:
                # Always use 0~1 as default range
                self.feature_ranges[feature] = [0.0, 1.0]

        self.log(f"Applied ranges for {len(self.feature_ranges)} features")

    def get_feature_bounds(self):
        """Get feature bounds for GA"""
        if not self.selected_features or not self.feature_ranges:
            raise ValueError("Features or feature ranges not set")

        lb = []
        ub = []

        for feature in self.selected_features:
            if feature in self.feature_ranges:
                lb.append(self.feature_ranges[feature][0])
                ub.append(self.feature_ranges[feature][1])
            else:
                # Always use 0~1 as default range
                lb.append(0.0)
                ub.append(1.0)

        return lb, ub

    def predict(self, X):
        """Make predictions using the model"""
        if self.model is None:
            raise ValueError("Model not loaded")

        if isinstance(X, list):
            X = np.array(X).reshape(1, -1)
        elif isinstance(X, np.ndarray) and len(X.shape) == 1:
            X = X.reshape(1, -1)

        if self.feature_scaler is not None:
            X = self.feature_scaler.transform(X)

        if X.shape[1] != len(self.selected_features):
            raise ValueError(f"Input features ({X.shape[1]}) don't match model ({len(self.selected_features)})")

        prediction = self.model.predict(X)[0]

        if self.target_scaler is not None:
            prediction = self.target_scaler.inverse_transform([[prediction]])[0][0]

        return prediction

    def get_model_info_summary(self):
        """Get model information summary"""
        summary = f"""
Model Information:
{'=' * 40}
Target Variable: {self.target_variable}
Number of Features: {len(self.selected_features)}
"""

        # Show feature names if available
        if self.original_feature_names:
            feature_list = ', '.join(self.original_feature_names[:5])
            if len(self.original_feature_names) > 5:
                feature_list += f"... (Total {len(self.original_feature_names)} features)"
            summary += f"Original Feature Names: {feature_list}\n"
        else:
            summary += f"Feature Names: Feature_1, Feature_2, ...\n"

        summary += f"""
Feature Scaling: {'Yes' if self.feature_scaler is not None else 'No'}
Target Scaling: {'Yes' if self.target_scaler is not None else 'No'}
Feature Statistics Available: {'Yes' if self.original_feature_stats else 'No'}
JSON Feature Info: {'Loaded' if self.json_feature_info else 'Not loaded'}
"""

        # Show key model parameters
        key_params = ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree']
        for param in key_params:
            if param in self.model_params:
                summary += f"{param}: {self.model_params[param]}\n"

        return summary

    def get_original_feature_names(self):
        """Get original feature names"""
        if self.original_feature_names and len(self.original_feature_names) == len(self.selected_features):
            return self.original_feature_names
        return self.selected_features

    def get_json_feature_names(self):
        """Get feature names from JSON"""
        return self.original_feature_names


# ====================== Genetic Algorithm Optimizer ======================
class GAOptimizer:
    """Genetic Algorithm Optimizer"""

    def __init__(self, model_loader, log_callback=None):
        self.model_loader = model_loader
        self.log_callback = log_callback
        self.optimization_history = []
        self.best_solution = None
        self.best_value = None
        self.ga_params = Config.GA_DEFAULT_PARAMS.copy()
        self.target_value = None
        self.optimization_type = 'maximize'
        self.multiple_run_results = []  # Store results from multiple runs
        self.toolbox = None
        self.population = None
        self.hof = None
        self.stats = None
        self.logbook = None
        
        # Initialize DEAP components
        self._init_deap()

    def log(self, message):
        """Log message"""
        if self.log_callback:
            self.log_callback(message)

    def _init_deap(self):
        """Initialize DEAP components"""
        # Create fitness class based on optimization type
        if not hasattr(creator, 'FitnessMax'):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, 'Individual'):
            creator.create("Individual", list, fitness=creator.FitnessMax)

    def set_ga_params(self, params):
        """Set GA parameters"""
        self.ga_params.update(params)

    def set_optimization_target(self, target_value=None, optimization_type='maximize'):
        """Set optimization target"""
        self.target_value = target_value
        self.optimization_type = optimization_type
        
        # Reinitialize DEAP with correct weights
        if self.optimization_type == 'minimize':
            if hasattr(creator, 'FitnessMax'):
                del creator.FitnessMax
            creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
        else:
            if hasattr(creator, 'FitnessMax'):
                del creator.FitnessMax
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            
        if not hasattr(creator, 'Individual'):
            creator.create("Individual", list, fitness=creator.FitnessMax)

    def evaluate(self, individual):
        """Evaluate an individual"""
        try:
            prediction = self.model_loader.predict(individual)
            
            if self.optimization_type == 'maximize':
                return prediction,
            elif self.optimization_type == 'minimize':
                return -prediction,
            elif self.optimization_type == 'target' and self.target_value is not None:
                # For target optimization, we want to minimize the difference
                difference = abs(prediction - self.target_value)
                # Convert to fitness (higher is better)
                fitness = 1.0 / (1.0 + difference)
                return fitness,
            else:
                return prediction,
                
        except Exception as e:
            self.log(f"Prediction error: {e}")
            return -1e10,

    def setup_toolbox(self):
        """Setup DEAP toolbox"""
        # Get feature bounds
        lb, ub = self.model_loader.get_feature_bounds()
        n_features = len(lb)
        
        # Create a new toolbox
        self.toolbox = base.Toolbox()
        
        # Register attribute generator
        def random_attribute(low, high):
            return random.uniform(low, high)
        
        # Register individual and population creation
        for i in range(n_features):
            self.toolbox.register(f"attr_{i}", random_attribute, lb[i], ub[i])
        
        attrs = [getattr(self.toolbox, f"attr_{i}") for i in range(n_features)]
        self.toolbox.register("individual", tools.initCycle, creator.Individual, 
                             attrs, n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Register evaluation function
        self.toolbox.register("evaluate", self.evaluate)
        
        # Register genetic operators
        self.toolbox.register("mate", tools.cxSimulatedBinary, eta=self.ga_params['eta'])
        self.toolbox.register("mutate", tools.mutPolynomialBounded, 
                             low=lb, up=ub, eta=self.ga_params['eta'], 
                             indpb=self.ga_params['mutation_prob'])
        self.toolbox.register("select", tools.selTournament, 
                             tournsize=self.ga_params['tournament_size'])
        
        # Create hall of fame
        self.hof = tools.HallOfFame(self.ga_params['hof_size'])
        
        # Setup statistics
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

    def run_optimization(self, run_number=1):
        """Run GA optimization"""
        try:
            self.log(f"Starting GA optimization (Run {run_number})...")
            
            # Setup toolbox
            self.setup_toolbox()
            
            # Create initial population
            self.population = self.toolbox.population(n=self.ga_params['population_size'])
            
            # Evaluate initial population
            fitnesses = list(map(self.toolbox.evaluate, self.population))
            for ind, fit in zip(self.population, fitnesses):
                ind.fitness.values = fit
            
            # Update hall of fame
            self.hof.update(self.population)
            
            # Statistics tracking
            record = self.stats.compile(self.population)
            self.logbook = tools.Logbook()
            self.logbook.record(gen=0, **record)
            
            start_time = time.time()
            
            # Main evolution loop
            for gen in range(1, self.ga_params['generations'] + 1):
                # Select the next generation individuals
                offspring = self.toolbox.select(self.population, len(self.population))
                
                # Clone the selected individuals
                offspring = list(map(self.toolbox.clone, offspring))
                
                # Apply crossover and mutation
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < self.ga_params['crossover_prob']:
                        self.toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values
                
                for mutant in offspring:
                    if random.random() < self.ga_params['mutation_prob']:
                        self.toolbox.mutate(mutant)
                        del mutant.fitness.values
                
                # Evaluate individuals with invalid fitness
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(self.toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                
                # Update hall of fame
                self.hof.update(offspring)
                
                # Elitism: keep the best individuals
                if self.ga_params['elitism_ratio'] > 0:
                    elites_size = int(self.ga_params['elitism_ratio'] * len(self.population))
                    elites = tools.selBest(self.population, elites_size)
                    offspring = elites + offspring[:len(offspring)-elites_size]
                
                # Replace population
                self.population[:] = offspring
                
                # Record statistics
                record = self.stats.compile(self.population)
                self.logbook.record(gen=gen, **record)
                
                # Log progress every 10 generations
                if gen % 10 == 0:
                    self.log(f"  Generation {gen}: Best = {record['max']:.6f}, Avg = {record['avg']:.6f}")
            
            elapsed_time = time.time() - start_time
            
            # Get best individual
            best_ind = self.hof.items[0]
            best_fitness = best_ind.fitness.values[0]
            
            # Convert fitness back to actual prediction value
            actual_prediction = self.model_loader.predict(best_ind)
            
            if self.optimization_type == 'maximize':
                best_value = best_fitness
            elif self.optimization_type == 'minimize':
                best_value = -best_fitness
            elif self.optimization_type == 'target':
                best_value = actual_prediction
            else:
                best_value = actual_prediction
            
            optimization_result = {
                'run_number': run_number,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'optimization_type': self.optimization_type,
                'target_value': self.target_value,
                'best_value': float(best_value),
                'actual_prediction': float(actual_prediction),
                'best_solution': [float(x) for x in best_ind],
                'elapsed_time': elapsed_time,
                'ga_params': self.ga_params.copy(),
                'generations': self.ga_params['generations'],
                'final_stats': record
            }
            
            self.optimization_history.append(optimization_result)
            self.best_solution = np.array(best_ind)
            self.best_value = best_value
            
            self.log(f"✓ GA optimization completed (Run {run_number})")
            self.log(f"  Best value: {best_value:.6f}")
            self.log(f"  Generations: {self.ga_params['generations']}")
            self.log(f"  Time: {elapsed_time:.2f}s")
            
            # Log final statistics
            self.log(f"  Final - Best: {record['max']:.6f}, Avg: {record['avg']:.6f}, Std: {record['std']:.6f}")
            
            return optimization_result
            
        except Exception as e:
            self.log(f"✗ GA optimization failed (Run {run_number}): {str(e)}")
            traceback.print_exc()
            raise

    def run_multiple_optimizations(self, n_runs=5):
        """Run multiple optimizations"""
        self.log(f"Starting {n_runs} GA runs...")
        self.multiple_run_results = []  # Reset results
        
        all_results = []
        best_overall = None
        best_overall_value = None
        
        for run in range(1, n_runs + 1):
            self.log(f"\nRun {run}/{n_runs}...")
            
            try:
                # Reset DEAP components for new run
                self._init_deap()
                
                # Run optimization with run number
                result = self.run_optimization(run_number=run)
                all_results.append(result)
                self.multiple_run_results.append(result)  # Store for later use
                
                # Update overall best
                if best_overall is None:
                    best_overall = result
                    best_overall_value = result['best_value']
                else:
                    if self.optimization_type == 'maximize':
                        if result['best_value'] > best_overall_value:
                            best_overall = result
                            best_overall_value = result['best_value']
                    elif self.optimization_type == 'minimize':
                        if result['best_value'] < best_overall_value:
                            best_overall = result
                            best_overall_value = result['best_value']
                    else:
                        if abs(result['best_value'] - self.target_value) < abs(best_overall_value - self.target_value):
                            best_overall = result
                            best_overall_value = result['best_value']
                
                self.log(f"  Current best: {best_overall_value:.6f}")
                
            except Exception as e:
                self.log(f"  Run {run} failed: {e}")
                continue
        
        if best_overall:
            self.best_solution = np.array(best_overall['best_solution'])
            self.best_value = best_overall['best_value']
        
        summary = {
            'total_runs': n_runs,
            'successful_runs': len(all_results),
            'best_overall': best_overall,
            'all_results': all_results
        }
        
        self.log(f"\nMultiple optimizations completed")
        self.log(f"  Successful: {len(all_results)}/{n_runs}")
        self.log(f"  Global best: {self.best_value:.6f}")
        
        return summary

    def get_solution_dict(self):
        """Get solution dictionary"""
        if self.best_solution is None or not self.model_loader.selected_features:
            return {}
        
        solution_dict = {}
        original_feature_names = self.model_loader.get_original_feature_names()
        
        for i, feature in enumerate(self.model_loader.selected_features):
            if i < len(original_feature_names):
                original_name = original_feature_names[i]
            else:
                original_name = feature
            solution_dict[original_name] = float(self.best_solution[i])
        
        return solution_dict
    
    def get_convergence_data(self):
        """Get convergence data for plotting"""
        if not self.logbook:
            return None
            
        generations = self.logbook.select("gen")
        best_values = self.logbook.select("max")
        avg_values = self.logbook.select("avg")
        
        return {
            'generations': generations,
            'best': best_values,
            'average': avg_values
        }


# ====================== Result Manager ======================
class ResultManager:
    """Result Manager"""

    def __init__(self, log_callback=None):
        self.log_callback = log_callback
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join(Config.GA_RESULTS_DIR, f"run_{self.timestamp}")
        os.makedirs(self.results_dir, exist_ok=True)

    def log(self, message):
        """Log message"""
        if self.log_callback:
            self.log_callback(message)

    def save_optimization_results(self, optimizer, model_loader, run_number=1):
        """Save optimization results for a single run"""
        try:
            self.log(f"Saving optimization results (Run {run_number})...")
            
            if optimizer.best_solution is None:
                self.log("Warning: No results to save")
                return None
            
            # Create subdirectory for this run
            run_dir = os.path.join(self.results_dir, f"run_{run_number}")
            os.makedirs(run_dir, exist_ok=True)
            
            # 1. Save best solution
            solution_dict = optimizer.get_solution_dict()
            original_feature_names = model_loader.get_original_feature_names()
            
            solution_df = pd.DataFrame({
                'Feature': list(solution_dict.keys()),
                'Optimal Value': list(solution_dict.values())
            })
            
            solution_path = os.path.join(run_dir, f"optimal_solution_run_{run_number}.csv")
            solution_df.to_csv(solution_path, index=False, encoding='utf-8-sig')
            self.log(f"✓ Optimal solution saved to: {solution_path}")
            
            # 2. Save optimization summary
            summary = {
                'timestamp': self.timestamp,
                'run_number': run_number,
                'optimization_type': optimizer.optimization_type,
                'target_value': optimizer.target_value,
                'best_value': float(optimizer.best_value),
                'actual_prediction': float(model_loader.predict(optimizer.best_solution)),
                'feature_count': len(solution_dict),
                'ga_parameters': optimizer.ga_params,
                'optimal_solution': solution_dict,
                'model_info': {
                    'target_variable': model_loader.target_variable,
                    'selected_features': model_loader.selected_features,
                    'original_feature_names': original_feature_names,
                    'feature_ranges': model_loader.feature_ranges
                }
            }
            
            summary_path = os.path.join(run_dir, f"optimization_summary_run_{run_number}.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            self.log(f"✓ Optimization summary saved to: {summary_path}")
            
            # 3. Save optimization history
            if optimizer.optimization_history:
                # Filter history for this specific run
                run_history = [result for result in optimizer.optimization_history if 
                              result.get('run_number') == run_number]
                if run_history:
                    history_df = pd.DataFrame(run_history)
                    history_path = os.path.join(run_dir, f"optimization_history_run_{run_number}.csv")
                    history_df.to_csv(history_path, index=False, encoding='utf-8-sig')
                    self.log(f"✓ Optimization history saved to: {history_path}")
            
            # 4. Save convergence data
            convergence_data = optimizer.get_convergence_data()
            if convergence_data:
                convergence_df = pd.DataFrame(convergence_data)
                convergence_path = os.path.join(run_dir, f"convergence_history_run_{run_number}.csv")
                convergence_df.to_csv(convergence_path, index=False, encoding='utf-8-sig')
                self.log(f"✓ Convergence history saved to: {convergence_path}")
            
            # 5. Create visualizations
            self.create_visualizations(optimizer, model_loader, solution_dict, run_dir, run_number)
            
            return summary
            
        except Exception as e:
            self.log(f"Result saving failed: {e}")
            traceback.print_exc()
            return None

    def save_multiple_run_results(self, optimizer, model_loader):
        """Save results from multiple runs"""
        try:
            self.log("Saving multiple run results...")
            
            if not optimizer.multiple_run_results:
                self.log("Warning: No multiple run results to save")
                return None
            
            # Save summary of all runs
            all_runs_summary = {
                'timestamp': self.timestamp,
                'total_runs': len(optimizer.multiple_run_results),
                'optimization_type': optimizer.optimization_type,
                'target_value': optimizer.target_value,
                'model_info': {
                    'target_variable': model_loader.target_variable,
                    'selected_features': model_loader.selected_features,
                    'original_feature_names': model_loader.get_original_feature_names(),
                    'feature_ranges': model_loader.feature_ranges
                },
                'runs': optimizer.multiple_run_results,
                'best_overall': optimizer.best_value if optimizer.best_value else None,
                'best_solution': optimizer.get_solution_dict() if optimizer.best_solution is not None else {}
            }
            
            summary_path = os.path.join(self.results_dir, "all_runs_summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(all_runs_summary, f, indent=2, ensure_ascii=False)
            self.log(f"✓ All runs summary saved to: {summary_path}")
            
            # Save all runs to CSV
            all_runs_df = pd.DataFrame(optimizer.multiple_run_results)
            all_runs_path = os.path.join(self.results_dir, "all_runs_results.csv")
            all_runs_df.to_csv(all_runs_path, index=False, encoding='utf-8-sig')
            self.log(f"✓ All runs results saved to: {all_runs_path}")
            
            # Create comparison visualization
            self.create_multiple_run_visualization(optimizer, model_loader)
            
            return all_runs_summary
            
        except Exception as e:
            self.log(f"Multiple run results saving failed: {e}")
            traceback.print_exc()
            return None

    def create_visualizations(self, optimizer, model_loader, solution_dict, run_dir, run_number):
        """Create visualization charts for a single run"""
        try:
            self.log(f"Creating visualization charts for Run {run_number}...")
            
            # 1. Feature importance chart
            if hasattr(model_loader.model, 'feature_importances_') and model_loader.selected_features:
                try:
                    plt.figure(figsize=(12, 8))
                    importance_scores = model_loader.model.feature_importances_
                    original_feature_names = model_loader.get_original_feature_names()
                    
                    features = []
                    for i in range(min(len(importance_scores), len(original_feature_names))):
                        features.append(original_feature_names[i])
                    
                    if len(features) < len(importance_scores):
                        for i in range(len(features), len(importance_scores)):
                            features.append(model_loader.selected_features[i])
                    
                    importance_df = pd.DataFrame({
                        'Feature': features,
                        'Importance': importance_scores[:len(features)]
                    })
                    importance_df = importance_df.sort_values('Importance', ascending=True)
                    
                    plt.barh(range(len(importance_df)), importance_df['Importance'], align='center')
                    plt.yticks(range(len(importance_df)), importance_df['Feature'])
                    plt.xlabel('Feature Importance')
                    plt.title(f'XGBoost Feature Importance - Run {run_number}')
                    plt.grid(True, alpha=0.3)
                    
                    importance_path = os.path.join(run_dir, f"feature_importance_run_{run_number}.png")
                    plt.tight_layout()
                    plt.savefig(importance_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    self.log(f"✓ Feature importance chart saved for Run {run_number}")
                except Exception as e:
                    self.log(f"Feature importance chart creation failed: {e}")
            
            # 2. Optimization solution chart
            if solution_dict:
                try:
                    plt.figure(figsize=(14, 6))
                    features = list(solution_dict.keys())
                    values = list(solution_dict.values())
                    
                    bars = plt.bar(features, values, alpha=0.7, color='steelblue')
                    plt.xlabel('Feature')
                    plt.ylabel('Optimal Value')
                    plt.title(f'Optimal Feature Values - Run {run_number}\nTarget: {model_loader.target_variable}')
                    plt.xticks(rotation=45, ha='right')
                    plt.grid(True, alpha=0.3, axis='y')
                    
                    for bar, value in zip(bars, values):
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width() / 2., height,
                                f'{value:.4f}', ha='center', va='bottom', fontsize=9)
                    
                    solution_path = os.path.join(run_dir, f"optimal_solution_chart_run_{run_number}.png")
                    plt.tight_layout()
                    plt.savefig(solution_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    self.log(f"✓ Optimal solution chart saved for Run {run_number}")
                except Exception as e:
                    self.log(f"Solution chart creation failed: {e}")
            
            # 3. Convergence plot
            convergence_data = optimizer.get_convergence_data()
            if convergence_data:
                try:
                    plt.figure(figsize=(10, 6))
                    plt.plot(convergence_data['generations'], convergence_data['best'], 
                            'b-', linewidth=2, label='Best Fitness')
                    plt.plot(convergence_data['generations'], convergence_data['average'], 
                            'r--', linewidth=1.5, label='Average Fitness')
                    plt.xlabel('Generation')
                    plt.ylabel('Fitness')
                    plt.title(f'GA Convergence Plot - Run {run_number}')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    convergence_path = os.path.join(run_dir, f"convergence_plot_run_{run_number}.png")
                    plt.tight_layout()
                    plt.savefig(convergence_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    self.log(f"✓ Convergence plot saved for Run {run_number}")
                except Exception as e:
                    self.log(f"Convergence plot creation failed: {e}")
            
            self.log(f"✓ Visualization charts created for Run {run_number}")
            
        except Exception as e:
            self.log(f"Visualization chart creation failed: {e}")

    def create_multiple_run_visualization(self, optimizer, model_loader):
        """Create visualization comparing multiple runs"""
        try:
            self.log("Creating multiple run comparison visualization...")
            
            if not optimizer.multiple_run_results or len(optimizer.multiple_run_results) < 2:
                self.log("Not enough runs for comparison")
                return
            
            # 1. Best value comparison across runs
            plt.figure(figsize=(10, 6))
            run_numbers = [result['run_number'] for result in optimizer.multiple_run_results]
            best_values = [result['best_value'] for result in optimizer.multiple_run_results]
            
            plt.plot(run_numbers, best_values, 'o-', linewidth=2, markersize=8)
            plt.xlabel('Run Number')
            plt.ylabel('Best Value')
            plt.title(f'Best Value Comparison Across Runs\nTarget: {model_loader.target_variable}')
            plt.grid(True, alpha=0.3)
            
            # Annotate each point
            for i, (run, value) in enumerate(zip(run_numbers, best_values)):
                plt.annotate(f'{value:.4f}', (run, value), textcoords="offset points",
                            xytext=(0, 10), ha='center', fontsize=9)
            
            comparison_path = os.path.join(self.results_dir, "multiple_runs_comparison.png")
            plt.tight_layout()
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.log(f"✓ Multiple runs comparison chart saved")
            
            # 2. Runtime comparison
            plt.figure(figsize=(10, 6))
            run_times = [result['elapsed_time'] for result in optimizer.multiple_run_results]
            
            plt.bar(run_numbers, run_times, alpha=0.7, color='green')
            plt.xlabel('Run Number')
            plt.ylabel('Runtime (seconds)')
            plt.title('Runtime Comparison Across Runs')
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add runtime labels on bars
            for i, (run, time_val) in enumerate(zip(run_numbers, run_times)):
                plt.text(run, time_val, f'{time_val:.2f}s', ha='center', va='bottom', fontsize=9)
            
            runtime_path = os.path.join(self.results_dir, "multiple_runs_runtime.png")
            plt.tight_layout()
            plt.savefig(runtime_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.log(f"✓ Runtime comparison chart saved")
            
            self.log("✓ Multiple run visualizations created")
            
        except Exception as e:
            self.log(f"Multiple run visualization creation failed: {e}")


# ====================== GUI Interface ======================
class GAXGBoostOptimizerGUI:
    """GA XGBoost Optimizer GUI"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Genetic Algorithm XGBoost Inverse Optimization System")

        # Set window size and position
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = min(1400, int(screen_width * 0.9))
        window_height = min(900, int(screen_height * 0.9))
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Create main scrollable frame
        self.main_canvas = tk.Canvas(self.root)
        self.scrollbar_v = ttk.Scrollbar(self.root, orient="vertical", command=self.main_canvas.yview)
        self.scrollbar_h = ttk.Scrollbar(self.root, orient="horizontal", command=self.main_canvas.xview)

        self.scrollable_frame = ttk.Frame(self.main_canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
        )

        self.main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.main_canvas.configure(yscrollcommand=self.scrollbar_v.set, xscrollcommand=self.scrollbar_h.set)

        self.main_canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar_v.grid(row=0, column=1, sticky="ns")
        self.scrollbar_h.grid(row=1, column=0, sticky="ew")

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        atexit.register(self.safe_cleanup)

        # Color configuration
        self.colors = {
            'primary': '#2c3e50',
            'secondary': '#3498db',
            'accent': '#e74c3c',
            'success': '#27ae60',
            'warning': '#f39c12',
            'background': '#f8f9fa',
            'text': '#2c3e50'
        }

        self.root.configure(bg=self.colors['background'])

        # Variables
        self.model_path = tk.StringVar()
        self.json_path = tk.StringVar()
        self.target_value = tk.DoubleVar(value=0.0)
        self.population_size = tk.IntVar(value=50)
        self.generations = tk.IntVar(value=100)
        self.crossover_prob = tk.DoubleVar(value=0.8)
        self.mutation_prob = tk.DoubleVar(value=0.2)
        self.tournament_size = tk.IntVar(value=3)
        self.eta = tk.DoubleVar(value=10.0)
        self.elitism_ratio = tk.DoubleVar(value=0.1)
        self.optimization_type = tk.StringVar(value="maximize")

        # State
        self.is_running = False
        self.model_loader = None
        self.optimizer = None
        self.result_manager = None
        self.threads = []

        # Initialize interface
        self.setup_styles()
        self.create_widgets()
        Config.ensure_directories()

    def setup_styles(self):
        """Setup styles"""
        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.style.configure('Title.TLabel',
                             font=('Segoe UI', 18, 'bold'),
                             background=self.colors['background'],
                             foreground=self.colors['primary'])

        self.style.configure('Section.TLabel',
                             font=('Segoe UI', 12, 'bold'),
                             background=self.colors['background'],
                             foreground=self.colors['secondary'])

    def create_widgets(self):
        """Create interface widgets"""
        main_frame = ttk.Frame(self.scrollable_frame, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # Title area
        title_frame = ttk.Frame(main_frame)
        title_frame.grid(row=0, column=0, columnspan=2, pady=(0, 20), sticky=(tk.W, tk.E))

        ttk.Label(title_frame, text="Genetic Algorithm XGBoost Inverse Optimization System",
                  style='Title.TLabel').grid(row=0, column=0, pady=5)

        ttk.Label(title_frame,
                  text="Use Genetic Algorithm to find input feature values for optimal output",
                  style='Section.TLabel').grid(row=1, column=0, pady=5)

        # Left configuration area
        config_frame = ttk.LabelFrame(main_frame, text="Model Configuration", padding="15")
        config_frame.grid(row=1, column=0, pady=10, padx=(0, 10), sticky=(tk.W, tk.E, tk.N, tk.S))
        config_frame.columnconfigure(1, weight=1)

        # Model loading
        row = 0
        ttk.Label(config_frame, text="XGBoost Model File:",
                  style='Section.TLabel').grid(row=row, column=0, sticky=tk.W, pady=5)

        model_frame = ttk.Frame(config_frame)
        model_frame.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        model_frame.columnconfigure(0, weight=1)

        ttk.Entry(model_frame, textvariable=self.model_path, width=30).grid(
            row=0, column=0, padx=(0, 5), sticky=(tk.W, tk.E))

        ttk.Button(model_frame, text="Browse...",
                   command=self.browse_model_file).grid(row=0, column=1, padx=5)

        ttk.Button(model_frame, text="Load Model",
                   command=self.load_model).grid(row=0, column=2, padx=5)

        # Feature info JSON loading
        row += 1
        ttk.Label(config_frame, text="Feature Info JSON File:",
                  style='Section.TLabel').grid(row=row, column=0, sticky=tk.W, pady=5)

        json_frame = ttk.Frame(config_frame)
        json_frame.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        json_frame.columnconfigure(0, weight=1)

        ttk.Entry(json_frame, textvariable=self.json_path, width=30).grid(
            row=0, column=0, padx=(0, 5), sticky=(tk.W, tk.E))

        ttk.Button(json_frame, text="Browse...",
                   command=self.browse_json_file).grid(row=0, column=1, padx=5)

        ttk.Button(json_frame, text="Load JSON",
                   command=self.load_feature_json).grid(row=0, column=2, padx=5)

        # Model information display
        row += 1
        ttk.Label(config_frame, text="Model Information:",
                  style='Section.TLabel').grid(row=row, column=0, sticky=tk.W, pady=5)

        self.model_info_text = scrolledtext.ScrolledText(config_frame, height=8,
                                                         font=('Consolas', 9),
                                                         bg='white', fg=self.colors['text'])
        self.model_info_text.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)

        # Feature range table
        row += 1
        ttk.Label(config_frame, text="Feature Range Settings:",
                  style='Section.TLabel').grid(row=row, column=0, sticky=tk.W, pady=5)

        table_frame = ttk.Frame(config_frame)
        table_frame.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        # Create Treeview with editing capability
        columns = ("Feature", "Min Value", "Max Value")
        self.feature_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=6)

        for col in columns:
            self.feature_tree.heading(col, text=col)
            if col == "Feature":
                self.feature_tree.column(col, width=150)
            else:
                self.feature_tree.column(col, width=100)

        # Add vertical scrollbar
        tree_scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.feature_tree.yview)
        self.feature_tree.configure(yscrollcommand=tree_scrollbar.set)

        self.feature_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        tree_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Bind double-click event for editing
        self.feature_tree.bind('<Double-1>', self.on_double_click)

        # Feature range buttons - only keep "Set All to [0,1]"
        row += 1
        range_buttons_frame = ttk.Frame(config_frame)
        range_buttons_frame.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)

        ttk.Button(range_buttons_frame, text="Set All to [0,1]",
                   command=self.set_all_ranges_0_1).grid(row=0, column=0, padx=5)

        # Right optimization configuration area
        opt_config_frame = ttk.LabelFrame(main_frame, text="Optimization Configuration", padding="15")
        opt_config_frame.grid(row=1, column=1, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        opt_config_frame.columnconfigure(1, weight=1)

        # Optimization type
        row = 0
        ttk.Label(opt_config_frame, text="Optimization Target:",
                  style='Section.TLabel').grid(row=row, column=0, sticky=tk.W, pady=5)

        opt_type_frame = ttk.Frame(opt_config_frame)
        opt_type_frame.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)

        ttk.Radiobutton(opt_type_frame, text="Maximize", variable=self.optimization_type,
                        value="maximize", command=self.update_target_entry_state).grid(row=0, column=0, padx=(0, 10))
        ttk.Radiobutton(opt_type_frame, text="Minimize", variable=self.optimization_type,
                        value="minimize", command=self.update_target_entry_state).grid(row=0, column=1, padx=(0, 10))
        ttk.Radiobutton(opt_type_frame, text="Target Value", variable=self.optimization_type,
                        value="target", command=self.update_target_entry_state).grid(row=0, column=2, padx=(0, 10))

        # Target value
        row += 1
        ttk.Label(opt_config_frame, text="Target Value:",
                  style='Section.TLabel').grid(row=row, column=0, sticky=tk.W, pady=5)

        self.target_entry = ttk.Entry(opt_config_frame, textvariable=self.target_value, width=15)
        self.target_entry.grid(row=row, column=1, sticky=tk.W, pady=5)

        # GA parameters
        row += 1
        ttk.Label(opt_config_frame, text="GA Parameters:",
                  style='Section.TLabel').grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5)

        row += 1
        ttk.Label(opt_config_frame, text="Population Size:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(opt_config_frame, from_=20, to=500, textvariable=self.population_size,
                    width=10).grid(row=row, column=1, sticky=tk.W, pady=2)

        row += 1
        ttk.Label(opt_config_frame, text="Generations:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(opt_config_frame, from_=10, to=500, textvariable=self.generations,
                    width=10).grid(row=row, column=1, sticky=tk.W, pady=2)

        row += 1
        ttk.Label(opt_config_frame, text="Crossover Prob:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(opt_config_frame, from_=0.1, to=1.0, increment=0.05, 
                   textvariable=self.crossover_prob, width=10).grid(row=row, column=1, sticky=tk.W, pady=2)

        row += 1
        ttk.Label(opt_config_frame, text="Mutation Prob:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(opt_config_frame, from_=0.01, to=0.5, increment=0.05,
                   textvariable=self.mutation_prob, width=10).grid(row=row, column=1, sticky=tk.W, pady=2)

        row += 1
        ttk.Label(opt_config_frame, text="Tournament Size:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(opt_config_frame, from_=2, to=10, textvariable=self.tournament_size,
                    width=10).grid(row=row, column=1, sticky=tk.W, pady=2)

        # Optimization control
        row += 1
        control_frame = ttk.Frame(opt_config_frame)
        control_frame.grid(row=row, column=0, columnspan=2, pady=15, sticky=(tk.W, tk.E))

        self.run_button = ttk.Button(control_frame, text="Start Optimization",
                                     command=self.start_optimization, state='disabled')
        self.run_button.grid(row=0, column=0, padx=5)

        ttk.Button(control_frame, text="Multiple Runs (5)",
                   command=self.start_multiple_optimizations).grid(row=0, column=1, padx=5)

        ttk.Button(control_frame, text="Open Results Directory",
                   command=self.open_results_dir).grid(row=0, column=2, padx=5)

        # Progress bar
        row += 1
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(opt_config_frame, variable=self.progress_var,
                                            length=400, mode='determinate')
        self.progress_bar.grid(row=row, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))

        # Status label
        self.status_label = ttk.Label(opt_config_frame, text="Ready")
        self.status_label.grid(row=row + 1, column=0, columnspan=2, pady=5)

        # Bottom results area
        results_frame = ttk.LabelFrame(main_frame, text="Optimization Results & Log", padding="15")
        results_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.rowconfigure(0, weight=1)
        results_frame.columnconfigure(0, weight=1)

        self.results_text = scrolledtext.ScrolledText(results_frame, height=15,
                                                      font=('Consolas', 10),
                                                      bg='white', fg=self.colors['text'])
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Bottom buttons
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.grid(row=3, column=0, columnspan=2, pady=10)

        ttk.Button(bottom_frame, text="Clear Results",
                   command=self.clear_results).grid(row=0, column=0, padx=5)

        ttk.Button(bottom_frame, text="Copy Results",
                   command=self.copy_results).grid(row=0, column=1, padx=5)

        ttk.Button(bottom_frame, text="Export Solution",
                   command=self.export_solution).grid(row=0, column=2, padx=5)

        ttk.Button(bottom_frame, text="Exit",
                   command=self.quit_program).grid(row=0, column=3, padx=5)

        # Usage instructions
        row += 1
        tip_frame = ttk.Frame(main_frame)
        tip_frame.grid(row=4, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))

        tip_text = """Usage Instructions:
1. First load XGBoost model file (.pkl)
2. Then load model info JSON file (contains feature names)
3. Feature ranges default to 0~1
4. Double-click on feature ranges in table to edit
5. Click "Set All to [0,1]" to reset all feature ranges
6. Click "Start Optimization" to run GA optimization
7. Click "Multiple Runs (5)" to run 5 optimizations (all results saved)
8. Results are saved in organized subfolders
9. Genetic Algorithm provides better diversity and avoids premature convergence"""

        tip_label = ttk.Label(tip_frame, text=tip_text, foreground="#666666",
                              font=('Segoe UI', 9), justify=tk.LEFT)
        tip_label.grid(row=0, column=0, sticky=tk.W)

        self.update_target_entry_state()

    def log(self, message):
        """Log message"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.results_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.results_text.see(tk.END)
        except (tk.TclError, AttributeError, RuntimeError):
            pass

    def update_status(self, message):
        """Update status"""
        try:
            self.status_label.configure(text=message)
        except (tk.TclError, AttributeError, RuntimeError):
            pass

    def update_progress(self, value):
        """Update progress"""
        try:
            self.progress_var.set(value)
        except (tk.TclError, AttributeError, RuntimeError):
            pass

    def update_target_entry_state(self):
        """Update target entry state"""
        if self.optimization_type.get() == "target":
            self.target_entry.configure(state='normal')
        else:
            self.target_entry.configure(state='disabled')

    def clear_results(self):
        """Clear results"""
        self.results_text.delete("1.0", tk.END)
        self.log("Results cleared")

    def copy_results(self):
        """Copy results to clipboard"""
        self.root.clipboard_clear()
        self.root.clipboard_append(self.results_text.get("1.0", tk.END))
        self.log("Results copied to clipboard")

    def browse_model_file(self):
        """Browse model file"""
        file_path = filedialog.askopenfilename(
            title="Select XGBoost Model File",
            filetypes=[("Model files", "*.pkl *.joblib *.model"), ("All files", "*.*")],
            initialdir=Config.PROJECT_ROOT
        )
        if file_path:
            self.model_path.set(file_path)
            self.log(f"Selected: {os.path.basename(file_path)}")

    def browse_json_file(self):
        """Browse JSON file"""
        file_path = filedialog.askopenfilename(
            title="Select Model Info JSON File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=Config.PROJECT_ROOT
        )
        if file_path:
            self.json_path.set(file_path)
            self.log(f"Selected JSON: {os.path.basename(file_path)}")

    def load_model(self):
        """Load model"""
        model_path = self.model_path.get()
        if not model_path:
            messagebox.showerror("Error", "Please select a model file")
            return

        if not os.path.exists(model_path):
            messagebox.showerror("Error", "Model file does not exist")
            return

        try:
            self.model_info_text.delete("1.0", tk.END)
            for item in self.feature_tree.get_children():
                self.feature_tree.delete(item)

            self.model_loader = XGBoostModelLoader(log_callback=self.log)
            success, message = self.model_loader.load_model(model_path)

            if success:
                self.log(f"✓ {message}")

                model_info = self.model_loader.get_model_info_summary()
                self.model_info_text.insert(tk.END, model_info)

                # Update feature table (using model features)
                self.update_feature_table()

                # If JSON already loaded, apply JSON info
                json_path = self.json_path.get()
                if json_path and os.path.exists(json_path):
                    self.load_feature_json()
                else:
                    # If no JSON, use model features
                    self.update_feature_table()

                self.run_button.configure(state='normal')

                self.log(f"Model loaded: {len(self.model_loader.selected_features)} features")

                original_names = self.model_loader.get_original_feature_names()
                if original_names:
                    self.log(
                        f"Original features: {', '.join(original_names[:5])}{'...' if len(original_names) > 5 else ''}")

                messagebox.showinfo("Success",
                                    f"Model loaded successfully!\nNumber of features: {len(self.model_loader.selected_features)}")
            else:
                self.log(f"✗ {message}")
                messagebox.showerror("Error", message)

        except Exception as e:
            self.log(f"✗ Model loading failed: {str(e)}")
            traceback.print_exc()
            messagebox.showerror("Error", f"Model loading failed: {str(e)}")

    def load_feature_json(self):
        """Load feature information JSON file"""
        json_path = self.json_path.get()
        if not json_path:
            messagebox.showerror("Error", "Please select a JSON file")
            return

        if not os.path.exists(json_path):
            messagebox.showerror("Error", "JSON file does not exist")
            return

        try:
            if not self.model_loader:
                self.model_loader = XGBoostModelLoader(log_callback=self.log)
                self.log("Created new model loader")

            success, message = self.model_loader.load_feature_info_json(json_path)

            if success:
                self.log(f"✓ {message}")

                # Update model information display
                model_info = self.model_loader.get_model_info_summary()
                self.model_info_text.delete("1.0", tk.END)
                self.model_info_text.insert(tk.END, model_info)

                # Update feature table - using feature names from JSON
                self.update_feature_table_with_json_names()

                self.log("Feature table updated with JSON information")
                messagebox.showinfo("Success", "Feature information loaded successfully!")
            else:
                self.log(f"✗ {message}")
                messagebox.showerror("Error", message)

        except Exception as e:
            error_msg = f"JSON loading failed: {str(e)}"
            self.log(f"✗ {error_msg}")
            traceback.print_exc()
            messagebox.showerror("Error", error_msg)

    def update_feature_table(self):
        """Update feature range table (using model features)"""
        for item in self.feature_tree.get_children():
            self.feature_tree.delete(item)

        if self.model_loader and self.model_loader.selected_features:
            original_feature_names = self.model_loader.get_original_feature_names()

            for i, feature in enumerate(self.model_loader.selected_features):
                # Prefer original feature names
                if i < len(original_feature_names):
                    feature_name = original_feature_names[i]
                else:
                    feature_name = feature

                # Get range values, default to 0~1
                if feature in self.model_loader.feature_ranges:
                    min_val, max_val = self.model_loader.feature_ranges[feature]
                else:
                    min_val, max_val = 0.0, 1.0

                # Format values to max 3 decimal places
                min_formatted = self._format_float(min_val)
                max_formatted = self._format_float(max_val)

                # Insert into table
                self.feature_tree.insert("", "end",
                                         values=(feature_name,
                                                 min_formatted,
                                                 max_formatted))

            self.log(f"✓ Feature table updated: {len(self.model_loader.selected_features)} features")

    def _format_float(self, value):
        """Format float value to max 3 decimal places"""
        try:
            # Convert to string with maximum 3 decimal places
            formatted = f"{float(value):.3f}"
            # Remove trailing zeros and decimal point if not needed
            if '.' in formatted:
                formatted = formatted.rstrip('0').rstrip('.')
            return formatted
        except:
            return str(value)

    def update_feature_table_with_json_names(self):
        """Update feature range table using feature names from JSON"""
        for item in self.feature_tree.get_children():
            self.feature_tree.delete(item)

        if self.model_loader:
            # Prefer feature names from JSON
            json_feature_names = self.model_loader.get_json_feature_names()

            if json_feature_names:
                # Use JSON feature names
                feature_names = json_feature_names
                self.log(f"Using JSON feature names: {len(feature_names)} features")

                # Show feature names from JSON
                for feature_name in feature_names:
                    # Get range values, default to 0~1
                    min_val, max_val = 0.0, 1.0

                    # Format values to max 3 decimal places
                    min_formatted = self._format_float(min_val)
                    max_formatted = self._format_float(max_val)

                    # Insert into table
                    self.feature_tree.insert("", "end",
                                             values=(feature_name,
                                                     min_formatted,
                                                     max_formatted))

                self.log(f"✓ Feature table updated: {len(feature_names)} features")
            else:
                self.log("Warning: No feature name information in JSON, using model features")
                self.update_feature_table()

    def on_double_click(self, event):
        """Handle double-click event to allow editing feature ranges"""
        # Get clicked row and column
        region = self.feature_tree.identify("region", event.x, event.y)
        if region != "cell":
            return

        # Get clicked row and column
        column = self.feature_tree.identify_column(event.x)
        row_id = self.feature_tree.identify_row(event.y)

        if not row_id:
            return

        # Get current value
        item = self.feature_tree.item(row_id)
        column_index = int(column.replace('#', '')) - 1

        # First column (feature name) not editable
        if column_index == 0:
            return

        # Get current value
        current_value = item['values'][column_index]

        # Create edit box
        x, y, width, height = self.feature_tree.bbox(row_id, column)

        # Create entry widget
        entry = ttk.Entry(self.feature_tree)
        entry.place(x=x, y=y, width=width, height=height)
        entry.insert(0, current_value)
        entry.select_range(0, tk.END)
        entry.focus()

        def save_edit(event=None):
            """Save edit"""
            new_value = entry.get()

            # Validate input is a valid number
            try:
                float_value = float(new_value)
                # Format to max 3 decimal places
                formatted_value = self._format_float(float_value)

                # Update value
                values = list(item['values'])
                values[column_index] = formatted_value
                self.feature_tree.item(row_id, values=values)

                # Log change
                feature_name = item['values'][0]
                if column_index == 1:
                    self.log(f"Modified min value for {feature_name} to {formatted_value}")
                else:
                    self.log(f"Modified max value for {feature_name} to {formatted_value}")
            except ValueError:
                self.log(f"Invalid numeric input: {new_value}")
                messagebox.showerror("Error", "Please enter a valid number")

            entry.destroy()

        def cancel_edit(event=None):
            """Cancel edit"""
            entry.destroy()

        # Bind events
        entry.bind("<Return>", save_edit)
        entry.bind("<Escape>", cancel_edit)
        entry.bind("<FocusOut>", lambda e: save_edit())

    def set_all_ranges_0_1(self):
        """Set all feature ranges to [0, 1]"""
        if not self.model_loader:
            messagebox.showwarning("Warning", "Please load a model first")
            return

        # Clear table
        for item in self.feature_tree.get_children():
            self.feature_tree.delete(item)

        # Determine which feature names to display
        if self.model_loader:
            # Prefer feature names from JSON
            json_feature_names = self.model_loader.get_json_feature_names()

            if json_feature_names:
                feature_names = json_feature_names
            else:
                # If no JSON feature names, use model features
                feature_names = self.model_loader.selected_features

            for feature_name in feature_names:
                self.feature_tree.insert("", "end", values=(feature_name, "0", "1"))

        self.log("✓ All feature ranges set to [0, 1]")

    def get_feature_ranges_from_table(self):
        """Get feature ranges from table"""
        if not self.model_loader or not self.model_loader.selected_features:
            return {}

        feature_ranges = {}

        for i, item in enumerate(self.feature_tree.get_children()):
            values = self.feature_tree.item(item, 'values')
            if len(values) >= 3:
                feature_name = values[0]

                try:
                    min_val = float(values[1])
                    max_val = float(values[2])

                    if min_val >= max_val:
                        self.log(f"Adjusting invalid range for {feature_name}")
                        max_val = min_val + 1.0

                    # Use feature index from model
                    if i < len(self.model_loader.selected_features):
                        model_feature = self.model_loader.selected_features[i]
                        feature_ranges[model_feature] = [min_val, max_val]
                    else:
                        # If index out of bounds, use feature name
                        feature_ranges[feature_name] = [min_val, max_val]
                except ValueError:
                    self.log(f"Invalid range for {feature_name}, using default")
                    if i < len(self.model_loader.selected_features):
                        model_feature = self.model_loader.selected_features[i]
                        feature_ranges[model_feature] = [0.0, 1.0]

        return feature_ranges

    def start_optimization(self):
        """Start optimization"""
        if self.is_running:
            messagebox.showwarning("Warning", "Optimization is already running, please wait")
            return

        if not self.model_loader or not self.model_loader.model:
            messagebox.showerror("Error", "Please load a model first")
            return

        feature_ranges = self.get_feature_ranges_from_table()
        if not feature_ranges:
            messagebox.showwarning("Warning", "Using default feature ranges [0,1]")

        self.model_loader.set_feature_ranges(feature_ranges)

        if self.optimization_type.get() == "target":
            try:
                target_val = float(self.target_value.get())
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid target value")
                return

        self.run_button.configure(state='disabled')
        self.is_running = True

        thread = threading.Thread(target=self.run_optimization)
        thread.daemon = True
        thread.start()
        self.threads.append(thread)

    def start_multiple_optimizations(self):
        """Start multiple optimizations"""
        if self.is_running:
            messagebox.showwarning("Warning", "Optimization is already running, please wait")
            return

        if not self.model_loader or not self.model_loader.model:
            messagebox.showerror("Error", "Please load a model first")
            return

        feature_ranges = self.get_feature_ranges_from_table()
        if not feature_ranges:
            messagebox.showwarning("Warning", "Using default feature ranges [0,1]")

        self.model_loader.set_feature_ranges(feature_ranges)

        if self.optimization_type.get() == "target":
            try:
                target_val = float(self.target_value.get())
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid target value")
                return

        self.run_button.configure(state='disabled')
        self.is_running = True

        thread = threading.Thread(target=self.run_multiple_optimizations)
        thread.daemon = True
        thread.start()
        self.threads.append(thread)

    def run_optimization(self):
        """Run optimization"""
        try:
            self.log("\n" + "=" * 70)
            self.log("Genetic Algorithm Optimization Started")
            self.log("=" * 70)

            optimization_type = self.optimization_type.get()
            target_value = self.target_value.get() if optimization_type == "target" else None

            ga_params = {
                'population_size': self.population_size.get(),
                'generations': self.generations.get(),
                'crossover_prob': self.crossover_prob.get(),
                'mutation_prob': self.mutation_prob.get(),
                'tournament_size': self.tournament_size.get(),
                'eta': self.eta.get(),
                'elitism_ratio': self.elitism_ratio.get()
            }

            self.update_status("Initializing...")
            self.update_progress(10)

            self.optimizer = GAOptimizer(self.model_loader, log_callback=self.log)
            self.optimizer.set_ga_params(ga_params)
            self.optimizer.set_optimization_target(target_value, optimization_type)

            self.update_status("Running GA...")
            self.update_progress(30)

            result = self.optimizer.run_optimization(run_number=1)

            self.update_status("Saving results...")
            self.update_progress(80)

            self.result_manager = ResultManager(log_callback=self.log)
            summary = self.result_manager.save_optimization_results(self.optimizer, self.model_loader, run_number=1)

            self.update_status("Completed!")
            self.update_progress(100)

            self.log("\n" + "=" * 70)
            self.log("Optimization Results")
            self.log("=" * 70)

            if summary:
                self.log(f"Type: {summary['optimization_type']}")
                if optimization_type == 'target':
                    self.log(f"Target: {summary['target_value']}")
                self.log(f"Best Value: {summary['best_value']:.6f}")
                self.log(f"Generations: {ga_params['generations']}")
                self.log(f"Population Size: {ga_params['population_size']}")
                self.log(f"Number of Features: {summary['feature_count']}")

                self.log(f"\nBest Solution:")
                for feature, value in summary['optimal_solution'].items():
                    self.log(f"  {feature}: {value:.6f}")

            self.log(f"\nResults saved to: {self.result_manager.results_dir}")

            if summary and 'optimal_solution' in summary:
                solution_text = "\n".join(
                    [f"{feature}: {value:.4f}" for feature, value in list(summary['optimal_solution'].items())[:5]])
                if len(summary['optimal_solution']) > 5:
                    solution_text += f"\n... (and {len(summary['optimal_solution']) - 5} more)"

                messagebox.showinfo("Completed",
                                    f"GA Optimization Completed!\n\n"
                                    f"Best Value: {summary['best_value']:.6f}\n"
                                    f"Generations: {ga_params['generations']}\n"
                                    f"Number of Features: {summary['feature_count']}\n"
                                    f"Best Solution:\n{solution_text}\n\n"
                                    f"Results saved to:\n{self.result_manager.results_dir}")
            else:
                messagebox.showinfo("Completed",
                                    f"GA Optimization Completed!\n\nBest Value: {result['best_value']:.6f}")

        except Exception as e:
            self.log(f"\n✗ Optimization failed: {str(e)}")
            traceback.print_exc()
            messagebox.showerror("Error", f"Optimization failed: {str(e)}")
        finally:
            self.run_button.configure(state='normal')
            self.is_running = False
            self.update_status("Ready")
            self.update_progress(0)
            self.threads = [t for t in self.threads if t.is_alive()]

    def run_multiple_optimizations(self):
        """Run multiple optimizations"""
        try:
            self.log("\n" + "=" * 70)
            self.log("Multiple GA Optimizations Started")
            self.log("=" * 70)

            optimization_type = self.optimization_type.get()
            target_value = self.target_value.get() if optimization_type == "target" else None

            ga_params = {
                'population_size': self.population_size.get(),
                'generations': self.generations.get(),
                'crossover_prob': self.crossover_prob.get(),
                'mutation_prob': self.mutation_prob.get(),
                'tournament_size': self.tournament_size.get(),
                'eta': self.eta.get(),
                'elitism_ratio': self.elitism_ratio.get()
            }

            self.update_status("Initializing...")
            self.update_progress(10)

            self.optimizer = GAOptimizer(self.model_loader, log_callback=self.log)
            self.optimizer.set_ga_params(ga_params)
            self.optimizer.set_optimization_target(target_value, optimization_type)

            self.update_status("Running multiple GA...")
            self.update_progress(30)

            summary = self.optimizer.run_multiple_optimizations(n_runs=5)

            self.update_status("Saving results...")
            self.update_progress(80)

            self.result_manager = ResultManager(log_callback=self.log)

            # Save individual run results (each in its own subfolder)
            for run_result in self.optimizer.multiple_run_results:
                run_number = run_result['run_number']
                # Temporarily set optimizer's best solution to this run's solution for saving
                original_solution = self.optimizer.best_solution
                original_value = self.optimizer.best_value

                self.optimizer.best_solution = np.array(run_result['best_solution'])
                self.optimizer.best_value = run_result['best_value']

                self.result_manager.save_optimization_results(self.optimizer, self.model_loader, run_number=run_number)

                # Restore original best solution
                self.optimizer.best_solution = original_solution
                self.optimizer.best_value = original_value

            # Save summary of all runs
            all_runs_summary = self.result_manager.save_multiple_run_results(self.optimizer, self.model_loader)

            self.update_status("Completed!")
            self.update_progress(100)

            self.log("\n" + "=" * 70)
            self.log("Multiple Optimization Results")
            self.log("=" * 70)

            self.log(f"Total Runs: {summary['total_runs']}")
            self.log(f"Successful Runs: {summary['successful_runs']}")

            if summary['best_overall']:
                best = summary['best_overall']
                self.log(f"\nGlobal Best Result:")
                self.log(f"  Type: {best['optimization_type']}")
                if optimization_type == 'target':
                    self.log(f"  Target: {best['target_value']}")
                self.log(f"  Best Value: {best['best_value']:.6f}")
                self.log(f"  Generations: {best['generations']}")

                solution_dict = self.optimizer.get_solution_dict()
                self.log(f"\n  Best Solution:")
                for feature, value in solution_dict.items():
                    self.log(f"    {feature}: {value:.6f}")

            self.log(f"\nResults saved to: {self.result_manager.results_dir}")
            self.log(f"Each run saved in separate subfolders: run_1, run_2, ...")

            if solution_dict:
                solution_text = "\n".join(
                    [f"{feature}: {value:.4f}" for feature, value in list(solution_dict.items())[:5]])
                if len(solution_dict) > 5:
                    solution_text += f"\n... (and {len(solution_dict) - 5} more)"

                messagebox.showinfo("Completed",
                                    f"Multiple GA Optimizations Completed!\n\n"
                                    f"Global Best: {summary['best_overall']['best_value']:.6f}\n"
                                    f"Successful Runs: {summary['successful_runs']}/{summary['total_runs']}\n"
                                    f"Best Solution:\n{solution_text}\n\n"
                                    f"Results saved to:\n{self.result_manager.results_dir}\n"
                                    f"Each run saved in separate subfolders.")
            else:
                messagebox.showinfo("Completed",
                                    f"Multiple GA Optimizations Completed!\n\n"
                                    f"Global Best: {summary['best_overall']['best_value']:.6f}\n"
                                    f"Successful Runs: {summary['successful_runs']}/{summary['total_runs']}")

        except Exception as e:
            self.log(f"\n✗ Multiple optimizations failed: {str(e)}")
            traceback.print_exc()
            messagebox.showerror("Error", f"Multiple optimizations failed: {str(e)}")
        finally:
            self.run_button.configure(state='normal')
            self.is_running = False
            self.update_status("Ready")
            self.update_progress(0)
            self.threads = [t for t in self.threads if t.is_alive()]

    def export_solution(self):
        """Export solution"""
        if not self.optimizer or self.optimizer.best_solution is None:
            messagebox.showerror("Error", "No solution to export")
            return

        try:
            solution_dict = self.optimizer.get_solution_dict()

            df = pd.DataFrame({
                'Feature': list(solution_dict.keys()),
                'Optimal Value': list(solution_dict.values())
            })

            if self.model_loader:
                df['Target Variable'] = self.model_loader.target_variable
                df['Optimization Type'] = self.optimizer.optimization_type
                if self.optimizer.optimization_type == 'target':
                    df['Target Value'] = self.optimizer.target_value
                df['Predicted Value'] = self.model_loader.predict(self.optimizer.best_solution)

            file_path = filedialog.asksaveasfilename(
                title="Save Solution",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")],
                initialfile="optimal_solution.csv"
            )

            if file_path:
                if file_path.endswith('.csv'):
                    df.to_csv(file_path, index=False, encoding='utf-8-sig')
                elif file_path.endswith('.xlsx'):
                    df.to_excel(file_path, index=False)

                self.log(f"✓ Solution exported to: {file_path}")
                messagebox.showinfo("Success", f"Solution exported to:\n{file_path}")

        except Exception as e:
            self.log(f"✗ Solution export failed: {str(e)}")
            traceback.print_exc()
            messagebox.showerror("Error", f"Solution export failed: {str(e)}")

    def open_results_dir(self):
        """Open results directory"""
        if hasattr(self, 'result_manager') and self.result_manager.results_dir:
            results_dir = self.result_manager.results_dir
        else:
            results_dir = Config.GA_RESULTS_DIR

        if os.path.exists(results_dir):
            try:
                if sys.platform == "win32":
                    os.startfile(results_dir)
                elif sys.platform == "darwin":
                    import subprocess
                    subprocess.call(["open", results_dir])
                else:
                    import subprocess
                    subprocess.call(["xdg-open", results_dir])
            except Exception as e:
                messagebox.showinfo("Info", f"Cannot open directory: {e}")
        else:
            messagebox.showinfo("Info", "Results directory does not exist")

    def on_closing(self):
        """Handle window closing"""
        if self.is_running:
            if not messagebox.askyesno("Confirm", "Optimization is running, are you sure you want to terminate?"):
                return
        self.quit_program()

    def quit_program(self):
        """Quit program"""
        self.safe_cleanup()
        sys.exit(0)

    def safe_cleanup(self):
        """Safe resource cleanup"""
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=0.5)

    def run(self):
        """Run GUI"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.safe_cleanup()
            sys.exit(0)
        except Exception as e:
            self.safe_cleanup()
            traceback.print_exc()
            sys.exit(1)


# ====================== Main Function ======================
def main():
    """Main function"""
    print("Genetic Algorithm XGBoost Inverse Optimization System")
    print("=" * 60)
    print(f"Project Root Directory: {Config.PROJECT_ROOT}")
    print(f"GA Results Directory: {Config.GA_RESULTS_DIR}")
    print("=" * 60)

    # Check dependencies
    try:
        import deap
        print("✓ deap library installed")
    except ImportError:
        print("✗ deap library not installed")
        print("Please install with: pip install deap")
        return

    try:
        import xgboost
        print("✓ xgboost library installed")
    except ImportError:
        print("✗ xgboost library not installed")
        print("Please install with: pip install xgboost")
        return

    Config.ensure_directories()
    print("All directories ready")

    # Set exception handling
    def handle_exception(exc_type, exc_value, exc_traceback):
        print(f"Uncaught exception: {exc_type.__name__}: {exc_value}")
        traceback.print_exception(exc_type, exc_value, exc_traceback)

    sys.excepthook = handle_exception

    # Start GUI
    app = GAXGBoostOptimizerGUI()
    app.run()


if __name__ == "__main__":
    main()