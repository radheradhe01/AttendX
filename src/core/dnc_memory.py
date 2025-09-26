"""
Differentiable Neural Computer (DNC) Implementation
Inspired by DeepMind's DNC architecture for scalable episodic and semantic memory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math

@dataclass
class DNCMemoryState:
    """State of the DNC memory system"""
    memory_matrix: torch.Tensor  # [batch_size, memory_size, word_size]
    read_weights: torch.Tensor   # [batch_size, num_read_heads, memory_size]
    write_weights: torch.Tensor  # [batch_size, memory_size]
    usage_vector: torch.Tensor   # [batch_size, memory_size]
    temporal_link_matrix: torch.Tensor  # [batch_size, memory_size, memory_size]
    precedence_vector: torch.Tensor    # [batch_size, memory_size]
    read_vectors: torch.Tensor   # [batch_size, num_read_heads, word_size]

class DNCMemorySystem(nn.Module):
    """
    Differentiable Neural Computer for episodic and semantic memory storage
    """
    
    def __init__(
        self,
        word_size: int = 64,
        memory_size: int = 256,
        num_read_heads: int = 4,
        num_write_heads: int = 1,
        controller_hidden_size: int = 256,
        controller_layers: int = 3
    ):
        super().__init__()
        
        self.word_size = word_size
        self.memory_size = memory_size
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        
        # Controller network (LSTM-based)
        self.controller = nn.LSTM(
            input_size=word_size + num_read_heads * word_size,
            hidden_size=controller_hidden_size,
            num_layers=controller_layers,
            batch_first=True
        )
        
        # Memory interface
        interface_size = num_read_heads * word_size + num_write_heads * word_size + 3 * num_write_heads + 5 * num_read_heads + 3
        self.interface_weights = nn.Linear(controller_hidden_size, interface_size)
        
        # Memory matrix initialization
        self.register_buffer('memory_matrix', torch.zeros(1, memory_size, word_size))
        
        # Episodic memory specific components
        self.episodic_encoder = nn.Linear(word_size, word_size)
        self.semantic_encoder = nn.Linear(word_size, word_size)
        
        # Attention mechanisms
        self.temporal_attention = nn.MultiheadAttention(word_size, num_heads=8)
        self.semantic_attention = nn.MultiheadAttention(word_size, num_heads=8)
        
    def forward(
        self, 
        input_vector: torch.Tensor, 
        prev_state: Optional[DNCMemoryState] = None,
        memory_type: str = "episodic"
    ) -> Tuple[torch.Tensor, DNCMemoryState]:
        """
        Forward pass through DNC memory system
        
        Args:
            input_vector: Input to be processed [batch_size, word_size]
            prev_state: Previous memory state
            memory_type: Type of memory ("episodic" or "semantic")
        
        Returns:
            output: Processed output [batch_size, word_size]
            new_state: Updated memory state
        """
        batch_size = input_vector.size(0)
        
        # Initialize state if None
        if prev_state is None:
            prev_state = self._initialize_state(batch_size)
        
        # Prepare controller input
        controller_input = self._prepare_controller_input(input_vector, prev_state)
        
        # Controller forward pass
        controller_output, (hidden_state, cell_state) = self.controller(controller_input)
        controller_output = controller_output.squeeze(1)  # Remove sequence dimension
        
        # Generate interface vector
        interface_vector = self.interface_weights(controller_output)
        
        # Parse interface vector
        parsed_interface = self._parse_interface_vector(interface_vector)
        
        # Memory operations
        new_state = self._memory_operations(parsed_interface, prev_state, memory_type)
        
        # Generate output
        output = self._generate_output(controller_output, new_state.read_vectors)
        
        return output, new_state
    
    def _initialize_state(self, batch_size: int) -> DNCMemoryState:
        """Initialize memory state"""
        device = next(self.parameters()).device
        
        return DNCMemoryState(
            memory_matrix=self.memory_matrix.expand(batch_size, -1, -1).clone(),
            read_weights=torch.zeros(batch_size, self.num_read_heads, self.memory_size, device=device),
            write_weights=torch.zeros(batch_size, self.memory_size, device=device),
            usage_vector=torch.zeros(batch_size, self.memory_size, device=device),
            temporal_link_matrix=torch.zeros(batch_size, self.memory_size, self.memory_size, device=device),
            precedence_vector=torch.zeros(batch_size, self.memory_size, device=device),
            read_vectors=torch.zeros(batch_size, self.num_read_heads, self.word_size, device=device)
        )
    
    def _prepare_controller_input(
        self, 
        input_vector: torch.Tensor, 
        prev_state: DNCMemoryState
    ) -> torch.Tensor:
        """Prepare input for controller"""
        # Concatenate input with previous read vectors
        read_vectors_flat = prev_state.read_vectors.view(
            prev_state.read_vectors.size(0), -1
        )
        controller_input = torch.cat([input_vector, read_vectors_flat], dim=-1)
        
        # Add sequence dimension for LSTM
        controller_input = controller_input.unsqueeze(1)
        
        return controller_input
    
    def _parse_interface_vector(self, interface_vector: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Parse interface vector into components"""
        parsed = {}
        idx = 0
        
        # Read keys
        parsed['read_keys'] = interface_vector[:, idx:idx + self.num_read_heads * self.word_size]
        idx += self.num_read_heads * self.word_size
        
        # Write key
        parsed['write_key'] = interface_vector[:, idx:idx + self.word_size]
        idx += self.word_size
        
        # Write strength
        parsed['write_strength'] = F.softplus(interface_vector[:, idx:idx + self.num_write_heads])
        idx += self.num_write_heads
        
        # Erase vector
        parsed['erase_vector'] = torch.sigmoid(interface_vector[:, idx:idx + self.num_write_heads * self.word_size])
        idx += self.num_write_heads * self.word_size
        
        # Write vector
        parsed['write_vector'] = interface_vector[:, idx:idx + self.num_write_heads * self.word_size]
        idx += self.num_write_heads * self.word_size
        
        # Read strengths
        parsed['read_strengths'] = F.softplus(interface_vector[:, idx:idx + self.num_read_heads])
        idx += self.num_read_heads
        
        # Read modes
        parsed['read_modes'] = torch.softmax(
            interface_vector[:, idx:idx + 3 * self.num_read_heads].view(
                -1, self.num_read_heads, 3
            ), dim=-1
        )
        idx += 3 * self.num_read_heads
        
        # Write gate
        parsed['write_gate'] = torch.sigmoid(interface_vector[:, idx:idx + self.num_write_heads])
        idx += self.num_write_heads
        
        # Allocation gate
        parsed['allocation_gate'] = torch.sigmoid(interface_vector[:, idx:idx + self.num_write_heads])
        idx += self.num_write_heads
        
        # Free gates
        parsed['free_gates'] = torch.sigmoid(interface_vector[:, idx:idx + self.num_read_heads])
        idx += self.num_read_heads
        
        return parsed
    
    def _memory_operations(
        self, 
        parsed_interface: Dict[str, torch.Tensor], 
        prev_state: DNCMemoryState,
        memory_type: str
    ) -> DNCMemoryState:
        """Perform memory read and write operations"""
        new_state = DNCMemoryState(
            memory_matrix=prev_state.memory_matrix.clone(),
            read_weights=prev_state.read_weights.clone(),
            write_weights=prev_state.write_weights.clone(),
            usage_vector=prev_state.usage_vector.clone(),
            temporal_link_matrix=prev_state.temporal_link_matrix.clone(),
            precedence_vector=prev_state.precedence_vector.clone(),
            read_vectors=prev_state.read_vectors.clone()
        )
        
        # Update usage vector
        new_state.usage_vector = self._update_usage_vector(
            parsed_interface, prev_state, new_state
        )
        
        # Compute write weights
        new_state.write_weights = self._compute_write_weights(
            parsed_interface, prev_state, new_state
        )
        
        # Write to memory
        new_state.memory_matrix = self._write_to_memory(
            parsed_interface, prev_state, new_state, memory_type
        )
        
        # Update temporal linkage
        new_state.temporal_link_matrix = self._update_temporal_linkage(
            parsed_interface, prev_state, new_state
        )
        
        # Update precedence vector
        new_state.precedence_vector = self._update_precedence_vector(
            parsed_interface, prev_state, new_state
        )
        
        # Compute read weights
        new_state.read_weights = self._compute_read_weights(
            parsed_interface, prev_state, new_state
        )
        
        # Read from memory
        new_state.read_vectors = self._read_from_memory(
            parsed_interface, prev_state, new_state
        )
        
        return new_state
    
    def _update_usage_vector(
        self, 
        parsed_interface: Dict[str, torch.Tensor], 
        prev_state: DNCMemoryState,
        new_state: DNCMemoryState
    ) -> torch.Tensor:
        """Update memory usage vector"""
        # Free gates determine how much memory is freed
        free_gates = parsed_interface['free_gates']  # [batch_size, num_read_heads]
        
        # Compute retention vector
        retention_vector = torch.prod(1 - free_gates.unsqueeze(-1) * prev_state.read_weights, dim=1)
        
        # Update usage vector
        usage_vector = prev_state.usage_vector * retention_vector
        
        return usage_vector
    
    def _compute_write_weights(
        self, 
        parsed_interface: Dict[str, torch.Tensor], 
        prev_state: DNCMemoryState,
        new_state: DNCMemoryState
    ) -> torch.Tensor:
        """Compute write weights using content-based and allocation-based addressing"""
        # Content-based addressing
        write_key = parsed_interface['write_key']  # [batch_size, word_size]
        write_strength = parsed_interface['write_strength']  # [batch_size, num_write_heads]
        
        # Compute content similarity
        content_similarity = F.cosine_similarity(
            write_key.unsqueeze(1),  # [batch_size, 1, word_size]
            prev_state.memory_matrix,  # [batch_size, memory_size, word_size]
            dim=-1
        )  # [batch_size, memory_size]
        
        # Apply write strength
        content_weights = F.softmax(write_strength.squeeze(-1).unsqueeze(-1) * content_similarity, dim=-1)
        
        # Allocation-based addressing
        allocation_weights = self._compute_allocation_weights(new_state.usage_vector)
        
        # Combine content and allocation weights
        allocation_gate = parsed_interface['allocation_gate']  # [batch_size, num_write_heads]
        write_gate = parsed_interface['write_gate']  # [batch_size, num_write_heads]
        
        write_weights = write_gate.squeeze(-1).unsqueeze(-1) * (
            allocation_gate.squeeze(-1).unsqueeze(-1) * allocation_weights +
            (1 - allocation_gate.squeeze(-1).unsqueeze(-1)) * content_weights
        )
        
        return write_weights
    
    def _compute_allocation_weights(self, usage_vector: torch.Tensor) -> torch.Tensor:
        """Compute allocation weights based on usage vector"""
        # Sort usage vector in ascending order
        sorted_usage, sorted_indices = torch.sort(usage_vector, dim=-1)
        
        # Compute allocation weights
        allocation_weights = torch.zeros_like(usage_vector)
        
        for i in range(self.memory_size):
            # Compute cumulative product
            cumprod = torch.cumprod(sorted_usage[:, :i+1], dim=-1)
            if i > 0:
                allocation_weights[:, sorted_indices[:, i]] = (
                    sorted_usage[:, i] * cumprod[:, i-1]
                )
            else:
                allocation_weights[:, sorted_indices[:, i]] = sorted_usage[:, i]
        
        return allocation_weights
    
    def _write_to_memory(
        self, 
        parsed_interface: Dict[str, torch.Tensor], 
        prev_state: DNCMemoryState,
        new_state: DNCMemoryState,
        memory_type: str
    ) -> torch.Tensor:
        """Write to memory matrix"""
        write_weights = new_state.write_weights  # [batch_size, memory_size]
        write_vector = parsed_interface['write_vector']  # [batch_size, num_write_heads * word_size]
        erase_vector = parsed_interface['erase_vector']  # [batch_size, num_write_heads * word_size]
        
        # Reshape write and erase vectors
        write_vector = write_vector.view(-1, self.num_write_heads, self.word_size)
        erase_vector = erase_vector.view(-1, self.num_write_heads, self.word_size)
        
        # Apply memory type specific encoding
        if memory_type == "episodic":
            write_vector = self.episodic_encoder(write_vector)
        elif memory_type == "semantic":
            write_vector = self.semantic_encoder(write_vector)
        
        # Erase and write operations
        memory_matrix = prev_state.memory_matrix.clone()
        
        for i in range(self.num_write_heads):
            # Erase
            memory_matrix = memory_matrix * (1 - write_weights.unsqueeze(-1) * erase_vector[:, i:i+1, :])
            
            # Write
            memory_matrix = memory_matrix + write_weights.unsqueeze(-1) * write_vector[:, i:i+1, :]
        
        return memory_matrix
    
    def _update_temporal_linkage(
        self, 
        parsed_interface: Dict[str, torch.Tensor], 
        prev_state: DNCMemoryState,
        new_state: DNCMemoryState
    ) -> torch.Tensor:
        """Update temporal linkage matrix"""
        write_weights = new_state.write_weights  # [batch_size, memory_size]
        precedence_vector = new_state.precedence_vector  # [batch_size, memory_size]
        
        # Update temporal link matrix
        temporal_link_matrix = prev_state.temporal_link_matrix.clone()
        
        # Compute new temporal links
        new_links = write_weights.unsqueeze(-1) * precedence_vector.unsqueeze(1)
        
        # Update matrix
        temporal_link_matrix = (1 - write_weights.unsqueeze(-1) - write_weights.unsqueeze(1)) * temporal_link_matrix + new_links
        
        return temporal_link_matrix
    
    def _update_precedence_vector(
        self, 
        parsed_interface: Dict[str, torch.Tensor], 
        prev_state: DNCMemoryState,
        new_state: DNCMemoryState
    ) -> torch.Tensor:
        """Update precedence vector"""
        write_weights = new_state.write_weights  # [batch_size, memory_size]
        prev_precedence = prev_state.precedence_vector  # [batch_size, memory_size]
        
        # Update precedence vector
        precedence_vector = (1 - write_weights.sum(dim=-1, keepdim=True)) * prev_precedence + write_weights
        
        return precedence_vector
    
    def _compute_read_weights(
        self, 
        parsed_interface: Dict[str, torch.Tensor], 
        prev_state: DNCMemoryState,
        new_state: DNCMemoryState
    ) -> torch.Tensor:
        """Compute read weights"""
        read_keys = parsed_interface['read_keys']  # [batch_size, num_read_heads * word_size]
        read_strengths = parsed_interface['read_strengths']  # [batch_size, num_read_heads]
        read_modes = parsed_interface['read_modes']  # [batch_size, num_read_heads, 3]
        
        # Reshape read keys
        read_keys = read_keys.view(-1, self.num_read_heads, self.word_size)
        
        # Content-based addressing
        content_similarity = F.cosine_similarity(
            read_keys,  # [batch_size, num_read_heads, word_size]
            new_state.memory_matrix.unsqueeze(1),  # [batch_size, 1, memory_size, word_size]
            dim=-1
        )  # [batch_size, num_read_heads, memory_size]
        
        content_weights = F.softmax(
            read_strengths.unsqueeze(-1) * content_similarity, 
            dim=-1
        )
        
        # Forward temporal addressing
        forward_weights = torch.bmm(
            prev_state.read_weights, 
            new_state.temporal_link_matrix
        )
        
        # Backward temporal addressing
        backward_weights = torch.bmm(
            prev_state.read_weights,
            new_state.temporal_link_matrix.transpose(-1, -2)
        )
        
        # Combine read modes
        read_weights = (
            read_modes[:, :, 0:1] * content_weights +
            read_modes[:, :, 1:2] * forward_weights +
            read_modes[:, :, 2:3] * backward_weights
        )
        
        return read_weights
    
    def _read_from_memory(
        self, 
        parsed_interface: Dict[str, torch.Tensor], 
        prev_state: DNCMemoryState,
        new_state: DNCMemoryState
    ) -> torch.Tensor:
        """Read from memory matrix"""
        read_weights = new_state.read_weights  # [batch_size, num_read_heads, memory_size]
        
        # Read vectors
        read_vectors = torch.bmm(
            read_weights,  # [batch_size, num_read_heads, memory_size]
            new_state.memory_matrix  # [batch_size, memory_size, word_size]
        )  # [batch_size, num_read_heads, word_size]
        
        return read_vectors
    
    def _generate_output(
        self, 
        controller_output: torch.Tensor, 
        read_vectors: torch.Tensor
    ) -> torch.Tensor:
        """Generate final output"""
        # Concatenate controller output with read vectors
        read_vectors_flat = read_vectors.view(read_vectors.size(0), -1)
        output = torch.cat([controller_output, read_vectors_flat], dim=-1)
        
        # Apply output transformation
        output = F.linear(output, torch.randn_like(output))
        
        return output
    
    def consolidate_memories(self, consolidation_threshold: float = 0.1):
        """Consolidate memories based on importance and recency"""
        # This would implement memory consolidation logic
        # For now, placeholder implementation
        pass
    
    def retrieve_memories(
        self, 
        query: torch.Tensor, 
        memory_state: DNCMemoryState,
        num_memories: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve relevant memories based on query
        
        Args:
            query: Query vector [batch_size, word_size]
            memory_state: Current memory state
            num_memories: Number of memories to retrieve
        
        Returns:
            retrieved_memories: Retrieved memory vectors
            attention_weights: Attention weights for retrieved memories
        """
        # Compute similarity between query and all memories
        similarities = F.cosine_similarity(
            query.unsqueeze(1),  # [batch_size, 1, word_size]
            memory_state.memory_matrix,  # [batch_size, memory_size, word_size]
            dim=-1
        )  # [batch_size, memory_size]
        
        # Get top-k most similar memories
        top_k_values, top_k_indices = torch.topk(similarities, num_memories, dim=-1)
        
        # Retrieve memory vectors
        batch_indices = torch.arange(query.size(0)).unsqueeze(-1).expand(-1, num_memories)
        retrieved_memories = memory_state.memory_matrix[batch_indices, top_k_indices]
        
        # Normalize attention weights
        attention_weights = F.softmax(top_k_values, dim=-1)
        
        return retrieved_memories, attention_weights

class EpisodicMemorySystem(DNCMemorySystem):
    """Specialized DNC for episodic memory"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Episodic-specific encoders
        self.experience_encoder = nn.Linear(self.word_size, self.word_size)
        self.temporal_encoder = nn.Linear(1, self.word_size)  # For timestamps
        self.context_encoder = nn.Linear(self.word_size, self.word_size)
        
    def encode_episode(
        self, 
        observation: torch.Tensor, 
        action: torch.Tensor, 
        reward: torch.Tensor,
        timestamp: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """Encode an episode for storage"""
        # Encode components
        obs_encoded = self.experience_encoder(observation)
        action_encoded = self.experience_encoder(action)
        reward_encoded = self.experience_encoder(reward)
        time_encoded = self.temporal_encoder(timestamp.unsqueeze(-1))
        context_encoded = self.context_encoder(context)
        
        # Combine into episode representation
        episode_vector = obs_encoded + action_encoded + reward_encoded + time_encoded + context_encoded
        
        return episode_vector

class SemanticMemorySystem(DNCMemorySystem):
    """Specialized DNC for semantic memory"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Semantic-specific encoders
        self.concept_encoder = nn.Linear(self.word_size, self.word_size)
        self.relation_encoder = nn.Linear(self.word_size, self.word_size)
        self.knowledge_encoder = nn.Linear(self.word_size, self.word_size)
        
    def encode_concept(
        self, 
        concept: torch.Tensor, 
        relations: List[torch.Tensor],
        confidence: torch.Tensor
    ) -> torch.Tensor:
        """Encode a concept with its relations"""
        # Encode concept
        concept_encoded = self.concept_encoder(concept)
        
        # Encode relations
        relation_encoded = torch.stack([
            self.relation_encoder(rel) for rel in relations
        ]).mean(dim=0)
        
        # Encode confidence
        confidence_encoded = self.knowledge_encoder(confidence)
        
        # Combine into semantic representation
        semantic_vector = concept_encoded + relation_encoded + confidence_encoded
        
        return semantic_vector

