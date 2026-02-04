
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
import sys
import os

# Import Action from System_03_ETARE
sys.path.append(os.path.abspath('01_Systems/System_03_ETARE'))
from ETARE_module import Action, DEVICE

class GeneticWeights:
    def __init__(self, input_size, device):
        self.input_weights = torch.empty(input_size, 128, device=device).uniform_(-0.5, 0.5)
        self.hidden_weights = torch.empty(128, 64, device=device).uniform_(-0.5, 0.5)
        self.output_weights = torch.empty(64, 6, device=device).uniform_(-0.5, 0.5)
        self.hidden_bias = torch.empty(128, device=device).uniform_(-0.5, 0.5)
        self.output_bias = torch.empty(6, device=device).uniform_(-0.5, 0.5)

class ETAREEnhanced:
    def __init__(self, input_size: int = 11, champion_features: int = 8):
        """
        Args:
            input_size: Total features (champion_features + quantum_features)
            champion_features: Number of features champions were trained on (8 for Redux)
        """
        self.input_size = input_size
        self.champion_features = champion_features
        self.weights = GeneticWeights(input_size, DEVICE)
        self.epsilon = 0.1
        self.learning_rate = 0.001

    def transfer_champion_weights(self, champion_data: dict):
        """
        Transfer weights from an 8-input Redux champion to an 11-input enhanced model.
        The first 8 inputs: rsi, macd, macd_signal, bb_upper, bb_lower, momentum, roc, atr
        The last 3 inputs: quantum features (compression_ratio, entropy, fused_confidence)
        """
        try:
            old_input_w = torch.tensor(champion_data['input_weights'], device=DEVICE, dtype=torch.float32)
            old_features = old_input_w.shape[0]

            print(f"Champion has {old_features} features, model expects {self.input_size} features")

            with torch.no_grad():
                # Transfer champion's technical feature weights
                self.weights.input_weights[:old_features, :] = old_input_w

                # Initialize quantum feature weights near zero (let them learn)
                # Small random init so they can contribute but don't disrupt champion behavior
                quantum_start = old_features
                self.weights.input_weights[quantum_start:, :] = torch.empty(
                    self.input_size - old_features, 128, device=DEVICE
                ).uniform_(-0.1, 0.1)

                # Transfer other layers directly
                self.weights.hidden_weights = torch.tensor(champion_data['hidden_weights'], device=DEVICE, dtype=torch.float32)
                self.weights.output_weights = torch.tensor(champion_data['output_weights'], device=DEVICE, dtype=torch.float32)
                self.weights.hidden_bias = torch.tensor(champion_data['hidden_bias'], device=DEVICE, dtype=torch.float32)
                self.weights.output_bias = torch.tensor(champion_data['output_bias'], device=DEVICE, dtype=torch.float32)

            print(f"✓ Transferred champion weights ({old_features} technical + {self.input_size - old_features} quantum features)")
        except Exception as e:
            print(f"Error transferring weights: {e}")

    def predict(self, state: np.ndarray) -> Tuple[Action, np.ndarray]:
        # state should be [1, 26]
        state_tensor = torch.FloatTensor(state).to(DEVICE)
        
        # Simple normalization
        mean = state_tensor.mean()
        std = state_tensor.std() + 1e-8
        state_tensor = (state_tensor - mean) / std

        # Forward pass
        with torch.no_grad():
            hidden = torch.tanh(torch.matmul(state_tensor, self.weights.input_weights) + self.weights.hidden_bias)
            hidden2 = torch.tanh(torch.matmul(hidden, self.weights.hidden_weights))
            output = torch.matmul(hidden2, self.weights.output_weights) + self.weights.output_bias
            
            probabilities = torch.softmax(output, dim=1)
            
        probs_np = probabilities.cpu().numpy()[0]

        if np.random.random() < self.epsilon:
            action = Action(np.random.randint(6))
        else:
            action = Action(np.argmax(probs_np))

        return action, probs_np

    def train_step(self, states, actions, rewards, next_states):
        """Minimal training step for fine-tuning the new quantum connections"""
        # This can be expanded to a full RL training loop
        pass
