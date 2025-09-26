"""
Model-Agnostic Meta-Learning (MAML) Implementation
Enables rapid adaptation while preserving continuity of self
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from dataclasses import dataclass
import copy

@dataclass
class MetaLearningState:
    """State for meta-learning process"""
    support_set: torch.Tensor
    query_set: torch.Tensor
    task_context: Dict
    adaptation_steps: int
    learning_rate: float

class MAMLAgent(nn.Module):
    """
    Model-Agnostic Meta-Learning agent for rapid adaptation
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        adaptation_lr: float = 0.01,
        meta_lr: float = 0.001,
        adaptation_steps: int = 5
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.adaptation_lr = adaptation_lr
        self.meta_lr = meta_lr
        self.adaptation_steps = adaptation_steps
        
        # Base network architecture
        self.base_network = self._build_network(input_dim, hidden_dim, output_dim, num_layers)
        
        # Meta-learning optimizer
        self.meta_optimizer = torch.optim.Adam(self.base_network.parameters(), lr=meta_lr)
        
        # Task-specific adaptation history
        self.adaptation_history = []
        
        # Self-model for meta-cognitive awareness
        self.self_model = SelfModelNetwork(hidden_dim)
        
    def _build_network(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> nn.Module:
        """Build the base network architecture"""
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, adapted_params: Optional[Dict] = None) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            x: Input tensor [batch_size, input_dim]
            adapted_params: Task-specific adapted parameters
        
        Returns:
            output: Network output [batch_size, output_dim]
        """
        if adapted_params is not None:
            # Use adapted parameters
            return self._forward_with_params(x, adapted_params)
        else:
            # Use base parameters
            return self.base_network(x)
    
    def _forward_with_params(self, x: torch.Tensor, params: Dict) -> torch.Tensor:
        """Forward pass with specific parameters"""
        # This would implement forward pass with custom parameters
        # For simplicity, using base network
        return self.base_network(x)
    
    def meta_train(
        self, 
        tasks: List[Dict], 
        num_meta_batches: int = 10
    ) -> Dict[str, float]:
        """
        Meta-training on a batch of tasks
        
        Args:
            tasks: List of task dictionaries containing support and query sets
            num_meta_batches: Number of meta-batches to process
        
        Returns:
            meta_loss: Meta-training loss
        """
        meta_losses = []
        
        for meta_batch in range(num_meta_batches):
            batch_tasks = np.random.choice(tasks, size=min(4, len(tasks)), replace=False)
            
            # Compute meta-gradient
            meta_gradients = self._compute_meta_gradients(batch_tasks)
            
            # Update meta-parameters
            self._update_meta_parameters(meta_gradients)
            
            # Compute meta-loss
            meta_loss = self._compute_meta_loss(batch_tasks)
            meta_losses.append(meta_loss.item())
        
        return {
            'meta_loss': np.mean(meta_losses),
            'meta_loss_std': np.std(meta_losses)
        }
    
    def _compute_meta_gradients(self, tasks: List[Dict]) -> Dict[str, torch.Tensor]:
        """Compute meta-gradients for a batch of tasks"""
        meta_gradients = {}
        
        # Initialize meta-gradients
        for name, param in self.base_network.named_parameters():
            meta_gradients[name] = torch.zeros_like(param)
        
        task_losses = []
        
        for task in tasks:
            # Adapt to task
            adapted_params, task_loss = self._adapt_to_task(task)
            task_losses.append(task_loss)
            
            # Compute gradients on query set
            query_loss = self._compute_query_loss(task, adapted_params)
            
            # Accumulate meta-gradients
            query_gradients = torch.autograd.grad(
                query_loss, 
                self.base_network.parameters(),
                create_graph=True,
                retain_graph=True
            )
            
            for i, (name, param) in enumerate(self.base_network.named_parameters()):
                meta_gradients[name] += query_gradients[i]
        
        # Average meta-gradients
        for name in meta_gradients:
            meta_gradients[name] /= len(tasks)
        
        return meta_gradients
    
    def _adapt_to_task(self, task: Dict) -> Tuple[Dict, torch.Tensor]:
        """Adapt to a specific task using gradient descent"""
        support_set = task['support_set']
        support_labels = task['support_labels']
        
        # Initialize adapted parameters
        adapted_params = {}
        for name, param in self.base_network.named_parameters():
            adapted_params[name] = param.clone()
        
        adaptation_losses = []
        
        for step in range(self.adaptation_steps):
            # Forward pass on support set
            support_output = self._forward_with_params(support_set, adapted_params)
            support_loss = F.mse_loss(support_output, support_labels)
            adaptation_losses.append(support_loss.item())
            
            # Compute gradients
            gradients = torch.autograd.grad(
                support_loss,
                adapted_params.values(),
                create_graph=True,
                retain_graph=True
            )
            
            # Update adapted parameters
            for i, (name, param) in enumerate(adapted_params.items()):
                adapted_params[name] = param - self.adaptation_lr * gradients[i]
        
        return adapted_params, adaptation_losses[-1]
    
    def _compute_query_loss(self, task: Dict, adapted_params: Dict) -> torch.Tensor:
        """Compute loss on query set with adapted parameters"""
        query_set = task['query_set']
        query_labels = task['query_labels']
        
        query_output = self._forward_with_params(query_set, adapted_params)
        query_loss = F.mse_loss(query_output, query_labels)
        
        return query_loss
    
    def _update_meta_parameters(self, meta_gradients: Dict[str, torch.Tensor]):
        """Update meta-parameters using meta-gradients"""
        for name, param in self.base_network.named_parameters():
            param.data = param.data - self.meta_lr * meta_gradients[name]
    
    def _compute_meta_loss(self, tasks: List[Dict]) -> torch.Tensor:
        """Compute meta-loss for monitoring"""
        total_loss = 0.0
        
        for task in tasks:
            adapted_params, _ = self._adapt_to_task(task)
            query_loss = self._compute_query_loss(task, adapted_params)
            total_loss += query_loss
        
        return total_loss / len(tasks)
    
    def rapid_adapt(
        self, 
        new_task: Dict, 
        adaptation_steps: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Rapidly adapt to a new task
        
        Args:
            new_task: New task with support and query sets
            adaptation_steps: Number of adaptation steps (optional)
        
        Returns:
            adapted_params: Task-specific adapted parameters
        """
        if adaptation_steps is None:
            adaptation_steps = self.adaptation_steps
        
        # Adapt to new task
        adapted_params, adaptation_loss = self._adapt_to_task(new_task)
        
        # Update self-model with adaptation experience
        self._update_self_model(new_task, adapted_params, adaptation_loss)
        
        # Store adaptation history
        self.adaptation_history.append({
            'task': new_task,
            'adapted_params': adapted_params,
            'loss': adaptation_loss,
            'timestamp': torch.tensor(time.time())
        })
        
        return adapted_params
    
    def _update_self_model(self, task: Dict, adapted_params: Dict, loss: torch.Tensor):
        """Update self-model based on adaptation experience"""
        # Extract task characteristics
        task_features = self._extract_task_features(task)
        
        # Update self-model
        self.self_model.update_model(task_features, loss)
    
    def _extract_task_features(self, task: Dict) -> torch.Tensor:
        """Extract features characterizing the task"""
        support_set = task['support_set']
        
        # Compute task statistics
        task_mean = support_set.mean(dim=0)
        task_std = support_set.std(dim=0)
        task_size = torch.tensor(support_set.size(0))
        
        # Combine features
        task_features = torch.cat([task_mean, task_std, task_size.unsqueeze(0)])
        
        return task_features
    
    def get_adaptation_confidence(self, task: Dict) -> float:
        """Get confidence in ability to adapt to a task"""
        task_features = self._extract_task_features(task)
        confidence = self.self_model.predict_confidence(task_features)
        
        return confidence.item()
    
    def explain_adaptation(self, task: Dict) -> Dict[str, str]:
        """Explain the adaptation process"""
        task_features = self._extract_task_features(task)
        
        explanation = {
            'task_complexity': self._assess_task_complexity(task_features),
            'adaptation_strategy': self._select_adaptation_strategy(task_features),
            'expected_performance': self._predict_performance(task_features),
            'similar_tasks': self._find_similar_tasks(task_features)
        }
        
        return explanation
    
    def _assess_task_complexity(self, task_features: torch.Tensor) -> str:
        """Assess task complexity"""
        # Simple heuristic based on task variance
        variance = task_features.var().item()
        
        if variance < 0.1:
            return "Low complexity - similar to seen tasks"
        elif variance < 0.5:
            return "Medium complexity - requires moderate adaptation"
        else:
            return "High complexity - significant adaptation needed"
    
    def _select_adaptation_strategy(self, task_features: torch.Tensor) -> str:
        """Select adaptation strategy based on task features"""
        # This would implement strategy selection logic
        return "Gradient-based adaptation with task-specific learning rate"
    
    def _predict_performance(self, task_features: torch.Tensor) -> str:
        """Predict expected performance on task"""
        confidence = self.self_model.predict_confidence(task_features)
        
        if confidence > 0.8:
            return "High expected performance"
        elif confidence > 0.5:
            return "Medium expected performance"
        else:
            return "Low expected performance - may need more adaptation"
    
    def _find_similar_tasks(self, task_features: torch.Tensor) -> str:
        """Find similar tasks from adaptation history"""
        if not self.adaptation_history:
            return "No similar tasks found"
        
        # Compute similarities with historical tasks
        similarities = []
        for history_item in self.adaptation_history:
            hist_features = self._extract_task_features(history_item['task'])
            similarity = F.cosine_similarity(
                task_features.unsqueeze(0), 
                hist_features.unsqueeze(0)
            ).item()
            similarities.append(similarity)
        
        max_similarity = max(similarities)
        if max_similarity > 0.7:
            return f"Similar to task with {max_similarity:.2f} similarity"
        else:
            return "No similar tasks found"

class SelfModelNetwork(nn.Module):
    """Self-model for meta-cognitive awareness"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Self-model network
        self.model_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Confidence output
            nn.Sigmoid()
        )
        
        # Adaptation experience memory
        self.experience_memory = []
        self.max_experiences = 1000
    
    def forward(self, task_features: torch.Tensor) -> torch.Tensor:
        """Forward pass through self-model"""
        return self.model_network(task_features)
    
    def update_model(self, task_features: torch.Tensor, adaptation_loss: torch.Tensor):
        """Update self-model based on adaptation experience"""
        # Store experience
        experience = {
            'task_features': task_features,
            'adaptation_loss': adaptation_loss,
            'timestamp': torch.tensor(time.time())
        }
        
        self.experience_memory.append(experience)
        
        # Limit memory size
        if len(self.experience_memory) > self.max_experiences:
            self.experience_memory = self.experience_memory[-self.max_experiences:]
        
        # Update model (simplified - would need proper training)
        self._update_model_weights(task_features, adaptation_loss)
    
    def _update_model_weights(self, task_features: torch.Tensor, adaptation_loss: torch.Tensor):
        """Update model weights based on new experience"""
        # This would implement proper model updating
        # For now, placeholder implementation
        pass
    
    def predict_confidence(self, task_features: torch.Tensor) -> torch.Tensor:
        """Predict confidence in ability to handle task"""
        return self.forward(task_features)
    
    def get_self_awareness_report(self) -> Dict[str, float]:
        """Generate self-awareness report"""
        if not self.experience_memory:
            return {'confidence': 0.0, 'experience_count': 0}
        
        # Compute average confidence
        confidences = []
        for experience in self.experience_memory:
            confidence = self.predict_confidence(experience['task_features'])
            confidences.append(confidence.item())
        
        return {
            'confidence': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'experience_count': len(self.experience_memory),
            'recent_performance': np.mean([exp['adaptation_loss'].item() for exp in self.experience_memory[-10:]])
        }

class ContinualLearningAgent(MAMLAgent):
    """MAML agent with continual learning capabilities"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Continual learning components
        self.task_memory = TaskMemory()
        self.catastrophic_forgetting_prevention = ForgettingPrevention()
        self.task_detector = TaskDetector()
        
    def continual_adapt(
        self, 
        new_data: torch.Tensor, 
        new_labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Continual adaptation to new data while preventing catastrophic forgetting
        
        Args:
            new_data: New data samples
            new_labels: New data labels
        
        Returns:
            adaptation_results: Results of adaptation process
        """
        # Detect if this is a new task
        is_new_task = self.task_detector.detect_new_task(new_data, new_labels)
        
        if is_new_task:
            # New task - use MAML adaptation
            new_task = {
                'support_set': new_data,
                'support_labels': new_labels,
                'query_set': new_data,  # Simplified
                'query_labels': new_labels
            }
            
            adapted_params = self.rapid_adapt(new_task)
            
            # Store task in memory
            self.task_memory.store_task(new_task, adapted_params)
            
        else:
            # Existing task - fine-tune
            adapted_params = self._fine_tune_existing_task(new_data, new_labels)
        
        # Prevent catastrophic forgetting
        self.catastrophic_forgetting_prevention.prevent_forgetting(
            self.task_memory.get_all_tasks()
        )
        
        return adapted_params
    
    def _fine_tune_existing_task(self, new_data: torch.Tensor, new_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Fine-tune on existing task"""
        # This would implement fine-tuning logic
        # For now, return base parameters
        adapted_params = {}
        for name, param in self.base_network.named_parameters():
            adapted_params[name] = param.clone()
        
        return adapted_params

class TaskMemory:
    """Memory system for storing task-specific information"""
    
    def __init__(self, max_tasks: int = 100):
        self.max_tasks = max_tasks
        self.tasks = []
        self.task_embeddings = []
    
    def store_task(self, task: Dict, adapted_params: Dict):
        """Store task and its adapted parameters"""
        task_info = {
            'task': task,
            'adapted_params': adapted_params,
            'timestamp': torch.tensor(time.time()),
            'performance': self._evaluate_task_performance(task, adapted_params)
        }
        
        self.tasks.append(task_info)
        
        # Store task embedding
        task_embedding = self._compute_task_embedding(task)
        self.task_embeddings.append(task_embedding)
        
        # Limit memory size
        if len(self.tasks) > self.max_tasks:
            self.tasks = self.tasks[-self.max_tasks:]
            self.task_embeddings = self.task_embeddings[-self.max_tasks:]
    
    def _compute_task_embedding(self, task: Dict) -> torch.Tensor:
        """Compute embedding for task"""
        support_set = task['support_set']
        return support_set.mean(dim=0)
    
    def _evaluate_task_performance(self, task: Dict, adapted_params: Dict) -> float:
        """Evaluate performance on task"""
        # This would implement performance evaluation
        return 0.8  # Placeholder
    
    def get_all_tasks(self) -> List[Dict]:
        """Get all stored tasks"""
        return self.tasks
    
    def find_similar_tasks(self, query_task: Dict, top_k: int = 5) -> List[Dict]:
        """Find similar tasks to query task"""
        query_embedding = self._compute_task_embedding(query_task)
        
        similarities = []
        for i, embedding in enumerate(self.task_embeddings):
            similarity = F.cosine_similarity(
                query_embedding.unsqueeze(0),
                embedding.unsqueeze(0)
            ).item()
            similarities.append((i, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k similar tasks
        similar_tasks = []
        for i, similarity in similarities[:top_k]:
            similar_tasks.append(self.tasks[i])
        
        return similar_tasks

class ForgettingPrevention:
    """Prevent catastrophic forgetting in continual learning"""
    
    def __init__(self):
        self.importance_weights = {}
        self.consolidation_threshold = 0.1
    
    def prevent_forgetting(self, stored_tasks: List[Dict]):
        """Prevent catastrophic forgetting"""
        # This would implement forgetting prevention strategies
        # Such as Elastic Weight Consolidation (EWC), Memory Aware Synapses (MAS), etc.
        pass
    
    def compute_importance_weights(self, task: Dict) -> Dict[str, torch.Tensor]:
        """Compute importance weights for parameters"""
        # This would implement importance weight computation
        return {}

class TaskDetector:
    """Detect when a new task is encountered"""
    
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self.seen_tasks = []
    
    def detect_new_task(self, new_data: torch.Tensor, new_labels: torch.Tensor) -> bool:
        """Detect if new data represents a new task"""
        if not self.seen_tasks:
            self.seen_tasks.append((new_data, new_labels))
            return True
        
        # Compute similarity with seen tasks
        new_task_embedding = new_data.mean(dim=0)
        
        for seen_data, seen_labels in self.seen_tasks:
            seen_embedding = seen_data.mean(dim=0)
            similarity = F.cosine_similarity(
                new_task_embedding.unsqueeze(0),
                seen_embedding.unsqueeze(0)
            ).item()
            
            if similarity > self.similarity_threshold:
                return False
        
        # New task detected
        self.seen_tasks.append((new_data, new_labels))
        return True

