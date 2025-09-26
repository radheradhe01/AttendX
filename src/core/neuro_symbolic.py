"""
Neuro-Symbolic Reasoning Layer
Implements Logic Tensor Networks (LTN) and Probabilistic Soft Logic (PSL) for explainable reasoning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import sympy as sp
from sympy import symbols, And, Or, Not, Implies, ForAll, Exists
import networkx as nx

@dataclass
class ReasoningState:
    """State of the neuro-symbolic reasoning system"""
    current_facts: Dict[str, torch.Tensor]
    active_rules: List[str]
    reasoning_trace: List[Dict]
    confidence_scores: Dict[str, float]
    attention_weights: torch.Tensor

class LogicTensorNetwork(nn.Module):
    """
    Logic Tensor Network implementation for neuro-symbolic reasoning
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        num_predicates: int = 10,
        num_constants: int = 100,
        max_arity: int = 3
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_predicates = num_predicates
        self.num_constants = num_constants
        self.max_arity = max_arity
        
        # Predicate networks
        self.predicates = nn.ModuleDict()
        self._initialize_predicates()
        
        # Constant embeddings
        self.constant_embeddings = nn.Embedding(num_constants, embedding_dim)
        
        # Reasoning attention mechanism
        self.reasoning_attention = nn.MultiheadAttention(embedding_dim, num_heads=8)
        
        # Confidence estimation
        self.confidence_estimator = nn.Linear(embedding_dim, 1)
        
        # Rule application network
        self.rule_network = RuleApplicationNetwork(embedding_dim)
        
    def _initialize_predicates(self):
        """Initialize predicate networks"""
        predicate_names = [
            'HasProperty', 'Causes', 'Enables', 'Prevents', 'SimilarTo',
            'PartOf', 'LocatedIn', 'TemporalBefore', 'TemporalAfter', 'AchievesGoal'
        ]
        
        for i, name in enumerate(predicate_names):
            # Create predicate network for different arities
            self.predicates[name] = nn.ModuleDict()
            for arity in range(1, self.max_arity + 1):
                self.predicates[name][f'arity_{arity}'] = PredicateNetwork(
                    embedding_dim, arity
                )
    
    def forward(
        self, 
        query: str, 
        facts: Dict[str, torch.Tensor],
        reasoning_state: Optional[ReasoningState] = None
    ) -> Tuple[torch.Tensor, ReasoningState, str]:
        """
        Perform neuro-symbolic reasoning on a query
        
        Args:
            query: Logical query string
            facts: Dictionary of known facts
            reasoning_state: Current reasoning state
        
        Returns:
            result: Reasoning result tensor
            new_state: Updated reasoning state
            explanation: Natural language explanation
        """
        if reasoning_state is None:
            reasoning_state = ReasoningState(
                current_facts=facts,
                active_rules=[],
                reasoning_trace=[],
                confidence_scores={},
                attention_weights=torch.zeros(self.embedding_dim)
            )
        
        # Parse query into logical form
        logical_form = self._parse_query(query)
        
        # Apply reasoning rules
        reasoning_result, reasoning_trace = self._apply_reasoning_rules(
            logical_form, reasoning_state
        )
        
        # Update reasoning state
        reasoning_state.reasoning_trace.extend(reasoning_trace)
        reasoning_state.current_facts.update(facts)
        
        # Generate explanation
        explanation = self._generate_explanation(reasoning_trace, reasoning_result)
        
        return reasoning_result, reasoning_state, explanation
    
    def _parse_query(self, query: str) -> Dict:
        """Parse natural language query into logical form"""
        # This would implement natural language to logic parsing
        # For now, simplified implementation
        
        logical_form = {
            'type': 'predicate',
            'predicate': 'HasProperty',
            'arguments': ['agent', 'consciousness'],
            'quantifiers': [],
            'connectives': []
        }
        
        return logical_form
    
    def _apply_reasoning_rules(
        self, 
        logical_form: Dict, 
        reasoning_state: ReasoningState
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """Apply reasoning rules to logical form"""
        reasoning_trace = []
        
        # Get predicate and arguments
        predicate_name = logical_form['predicate']
        arguments = logical_form['arguments']
        
        # Look up predicate network
        arity = len(arguments)
        predicate_network = self.predicates[predicate_name][f'arity_{arity}']
        
        # Encode arguments
        argument_embeddings = []
        for arg in arguments:
            if arg in reasoning_state.current_facts:
                arg_embedding = reasoning_state.current_facts[arg]
            else:
                # Use constant embedding
                arg_id = self._get_constant_id(arg)
                arg_embedding = self.constant_embeddings(arg_id)
            
            argument_embeddings.append(arg_embedding)
        
        # Apply predicate
        result = predicate_network(torch.stack(argument_embeddings))
        
        # Compute confidence
        confidence = torch.sigmoid(self.confidence_estimator(result))
        
        # Record reasoning step
        reasoning_step = {
            'rule': f"{predicate_name}({', '.join(arguments)})",
            'result': result,
            'confidence': confidence.item(),
            'timestamp': torch.tensor(time.time())
        }
        reasoning_trace.append(reasoning_step)
        
        # Update confidence scores
        reasoning_state.confidence_scores[f"{predicate_name}({', '.join(arguments)})"] = confidence.item()
        
        return result, reasoning_trace
    
    def _get_constant_id(self, constant: str) -> int:
        """Get constant ID for embedding lookup"""
        # Simple hash-based mapping
        return hash(constant) % self.num_constants
    
    def _generate_explanation(
        self, 
        reasoning_trace: List[Dict], 
        final_result: torch.Tensor
    ) -> str:
        """Generate natural language explanation of reasoning"""
        explanation_parts = []
        
        for step in reasoning_trace:
            rule = step['rule']
            confidence = step['confidence']
            
            if confidence > 0.7:
                explanation_parts.append(f"Applied rule '{rule}' with high confidence ({confidence:.2f})")
            elif confidence > 0.4:
                explanation_parts.append(f"Applied rule '{rule}' with medium confidence ({confidence:.2f})")
            else:
                explanation_parts.append(f"Applied rule '{rule}' with low confidence ({confidence:.2f})")
        
        explanation = "Reasoning process: " + "; ".join(explanation_parts)
        
        return explanation

class PredicateNetwork(nn.Module):
    """Network for computing predicate satisfiability"""
    
    def __init__(self, embedding_dim: int, arity: int):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.arity = arity
        
        # Multi-layer network for predicate computation
        self.predicate_network = nn.Sequential(
            nn.Linear(embedding_dim * arity, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, argument_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute predicate satisfiability
        
        Args:
            argument_embeddings: Tensor of shape [arity, embedding_dim]
        
        Returns:
            satisfiability: Scalar tensor [0, 1]
        """
        # Flatten argument embeddings
        flattened = argument_embeddings.view(-1)
        
        # Apply predicate network
        satisfiability = self.predicate_network(flattened)
        
        return satisfiability

class RuleApplicationNetwork(nn.Module):
    """Network for applying logical rules"""
    
    def __init__(self, embedding_dim: int):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Rule application network
        self.rule_network = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim * 2),  # premise1, premise2, conclusion
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self, 
        premise1: torch.Tensor, 
        premise2: torch.Tensor, 
        conclusion: torch.Tensor
    ) -> torch.Tensor:
        """Apply logical rule to premises to derive conclusion"""
        # Concatenate premises and conclusion
        combined = torch.cat([premise1, premise2, conclusion], dim=-1)
        
        # Apply rule network
        rule_applicability = self.rule_network(combined)
        
        return rule_applicability

class ProbabilisticSoftLogic(nn.Module):
    """
    Probabilistic Soft Logic implementation for uncertain reasoning
    """
    
    def __init__(self, num_rules: int = 50, embedding_dim: int = 64):
        super().__init__()
        
        self.num_rules = num_rules
        self.embedding_dim = embedding_dim
        
        # Rule weights (learnable)
        self.rule_weights = nn.Parameter(torch.randn(num_rules))
        
        # Rule embeddings
        self.rule_embeddings = nn.Embedding(num_rules, embedding_dim)
        
        # Grounding network
        self.grounding_network = GroundingNetwork(embedding_dim)
        
        # Inference network
        self.inference_network = InferenceNetwork(embedding_dim)
        
    def forward(
        self, 
        query_atoms: List[str], 
        evidence_atoms: Dict[str, torch.Tensor],
        rules: List[str]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Perform probabilistic soft logic inference
        
        Args:
            query_atoms: List of query atoms
            evidence_atoms: Dictionary of evidence atoms and their values
            rules: List of logical rules
        
        Returns:
            query_results: Results for query atoms
            atom_probabilities: Probability distribution over all atoms
        """
        # Ground rules
        grounded_rules = self._ground_rules(rules, evidence_atoms)
        
        # Compute rule satisfactions
        rule_satisfactions = self._compute_rule_satisfactions(grounded_rules, evidence_atoms)
        
        # Weighted combination
        weighted_satisfactions = rule_satisfactions * self.rule_weights
        
        # Inference
        atom_probabilities = self._perform_inference(weighted_satisfactions, evidence_atoms)
        
        # Extract query results
        query_results = torch.stack([
            atom_probabilities.get(atom, torch.tensor(0.5)) 
            for atom in query_atoms
        ])
        
        return query_results, atom_probabilities
    
    def _ground_rules(self, rules: List[str], evidence_atoms: Dict[str, torch.Tensor]) -> List[Dict]:
        """Ground logical rules with evidence"""
        grounded_rules = []
        
        for i, rule in enumerate(rules):
            grounded_rule = {
                'rule_id': i,
                'rule_text': rule,
                'rule_embedding': self.rule_embeddings(torch.tensor(i)),
                'groundings': self._find_groundings(rule, evidence_atoms)
            }
            grounded_rules.append(grounded_rule)
        
        return grounded_rules
    
    def _find_groundings(self, rule: str, evidence_atoms: Dict[str, torch.Tensor]) -> List[Dict]:
        """Find groundings for a rule"""
        # Simplified grounding - would need proper logical parsing
        groundings = []
        
        # Extract atoms from rule
        atoms = self._extract_atoms(rule)
        
        for atom in atoms:
            if atom in evidence_atoms:
                grounding = {
                    'atom': atom,
                    'value': evidence_atoms[atom],
                    'confidence': torch.tensor(1.0)
                }
                groundings.append(grounding)
        
        return groundings
    
    def _extract_atoms(self, rule: str) -> List[str]:
        """Extract atoms from rule string"""
        # Simplified atom extraction
        atoms = []
        words = rule.split()
        
        for word in words:
            if word.isalpha() and len(word) > 1:
                atoms.append(word)
        
        return atoms
    
    def _compute_rule_satisfactions(
        self, 
        grounded_rules: List[Dict], 
        evidence_atoms: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute satisfactions for grounded rules"""
        satisfactions = []
        
        for rule in grounded_rules:
            # Compute satisfaction for this rule
            satisfaction = self._compute_rule_satisfaction(rule, evidence_atoms)
            satisfactions.append(satisfaction)
        
        return torch.stack(satisfactions)
    
    def _compute_rule_satisfaction(
        self, 
        rule: Dict, 
        evidence_atoms: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute satisfaction for a single rule"""
        # Simplified satisfaction computation
        # Would need proper logical evaluation
        
        groundings = rule['groundings']
        if not groundings:
            return torch.tensor(0.5)  # Neutral satisfaction
        
        # Average satisfaction over groundings
        satisfactions = []
        for grounding in groundings:
            atom_value = grounding['value']
            confidence = grounding['confidence']
            satisfaction = atom_value * confidence
            satisfactions.append(satisfaction)
        
        return torch.mean(torch.stack(satisfactions))
    
    def _perform_inference(
        self, 
        weighted_satisfactions: torch.Tensor, 
        evidence_atoms: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Perform probabilistic inference"""
        # Simplified inference - would need proper PSL inference
        atom_probabilities = {}
        
        # Initialize with evidence
        atom_probabilities.update(evidence_atoms)
        
        # Update based on rule satisfactions
        for i, satisfaction in enumerate(weighted_satisfactions):
            # This would implement proper PSL inference
            # For now, simplified update
            atom_probabilities[f'rule_{i}'] = torch.sigmoid(satisfaction)
        
        return atom_probabilities

class GroundingNetwork(nn.Module):
    """Network for grounding logical expressions"""
    
    def __init__(self, embedding_dim: int):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Grounding network
        self.grounding_network = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, logical_expression: torch.Tensor) -> torch.Tensor:
        """Ground logical expression"""
        return self.grounding_network(logical_expression)

class InferenceNetwork(nn.Module):
    """Network for probabilistic inference"""
    
    def __init__(self, embedding_dim: int):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Inference network
        self.inference_network = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, rule_satisfactions: torch.Tensor) -> torch.Tensor:
        """Perform inference based on rule satisfactions"""
        return self.inference_network(rule_satisfactions)

class MetaCognitiveReasoner(nn.Module):
    """
    Meta-cognitive reasoning system for self-reflection and explanation
    """
    
    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Self-model for meta-cognitive awareness
        self.self_model = SelfModelNetwork(embedding_dim)
        
        # Explanation generator
        self.explanation_generator = ExplanationGenerator(embedding_dim)
        
        # Confidence estimator
        self.confidence_estimator = ConfidenceEstimator(embedding_dim)
        
        # Attention monitor
        self.attention_monitor = AttentionMonitor(embedding_dim)
        
    def reflect_on_reasoning(
        self, 
        reasoning_process: List[Dict], 
        final_result: torch.Tensor
    ) -> Dict[str, Union[str, float]]:
        """
        Reflect on the reasoning process
        
        Args:
            reasoning_process: List of reasoning steps
            final_result: Final reasoning result
        
        Returns:
            reflection: Meta-cognitive reflection
        """
        reflection = {}
        
        # Analyze reasoning quality
        reasoning_quality = self._analyze_reasoning_quality(reasoning_process)
        reflection['reasoning_quality'] = reasoning_quality
        
        # Estimate confidence
        confidence = self.confidence_estimator(final_result)
        reflection['confidence'] = confidence.item()
        
        # Generate explanation
        explanation = self.explanation_generator(reasoning_process, final_result)
        reflection['explanation'] = explanation
        
        # Monitor attention
        attention_pattern = self.attention_monitor(reasoning_process)
        reflection['attention_pattern'] = attention_pattern
        
        # Self-assessment
        self_assessment = self._self_assess(reasoning_process, final_result)
        reflection['self_assessment'] = self_assessment
        
        return reflection
    
    def _analyze_reasoning_quality(self, reasoning_process: List[Dict]) -> float:
        """Analyze the quality of reasoning process"""
        if not reasoning_process:
            return 0.0
        
        # Analyze consistency
        consistency_score = self._compute_consistency(reasoning_process)
        
        # Analyze completeness
        completeness_score = self._compute_completeness(reasoning_process)
        
        # Analyze efficiency
        efficiency_score = self._compute_efficiency(reasoning_process)
        
        # Combine scores
        quality_score = (consistency_score + completeness_score + efficiency_score) / 3
        
        return quality_score
    
    def _compute_consistency(self, reasoning_process: List[Dict]) -> float:
        """Compute consistency score for reasoning process"""
        # Check for logical contradictions
        contradictions = 0
        total_steps = len(reasoning_process)
        
        for i, step in enumerate(reasoning_process):
            for j, other_step in enumerate(reasoning_process[i+1:], i+1):
                if self._are_contradictory(step, other_step):
                    contradictions += 1
        
        consistency_score = 1.0 - (contradictions / max(1, total_steps * (total_steps - 1) / 2))
        return consistency_score
    
    def _are_contradictory(self, step1: Dict, step2: Dict) -> bool:
        """Check if two reasoning steps are contradictory"""
        # Simplified contradiction detection
        return False  # Placeholder
    
    def _compute_completeness(self, reasoning_process: List[Dict]) -> float:
        """Compute completeness score for reasoning process"""
        # Check if all necessary steps were taken
        return 0.8  # Placeholder
    
    def _compute_efficiency(self, reasoning_process: List[Dict]) -> float:
        """Compute efficiency score for reasoning process"""
        # Check if reasoning was efficient (not too many redundant steps)
        return 0.7  # Placeholder
    
    def _self_assess(self, reasoning_process: List[Dict], final_result: torch.Tensor) -> str:
        """Perform self-assessment of reasoning"""
        quality = self._analyze_reasoning_quality(reasoning_process)
        
        if quality > 0.8:
            return "High-quality reasoning with strong logical consistency"
        elif quality > 0.6:
            return "Good reasoning with minor inconsistencies"
        elif quality > 0.4:
            return "Adequate reasoning with some issues"
        else:
            return "Poor reasoning with significant problems"

class SelfModelNetwork(nn.Module):
    """Self-model for meta-cognitive awareness"""
    
    def __init__(self, embedding_dim: int):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Self-model network
        self.self_model = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(self, reasoning_state: torch.Tensor) -> torch.Tensor:
        """Generate self-model representation"""
        return self.self_model(reasoning_state)

class ExplanationGenerator(nn.Module):
    """Generate natural language explanations"""
    
    def __init__(self, embedding_dim: int):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Explanation network
        self.explanation_network = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )
        
    def forward(self, reasoning_process: List[Dict], final_result: torch.Tensor) -> str:
        """Generate explanation for reasoning process"""
        # Simplified explanation generation
        explanation = f"Applied {len(reasoning_process)} reasoning steps to reach conclusion with confidence {final_result.item():.2f}"
        return explanation

class ConfidenceEstimator(nn.Module):
    """Estimate confidence in reasoning results"""
    
    def __init__(self, embedding_dim: int):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Confidence network
        self.confidence_network = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, result: torch.Tensor) -> torch.Tensor:
        """Estimate confidence in result"""
        return self.confidence_network(result)

class AttentionMonitor(nn.Module):
    """Monitor attention patterns during reasoning"""
    
    def __init__(self, embedding_dim: int):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Attention monitoring network
        self.attention_network = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(self, reasoning_process: List[Dict]) -> Dict[str, float]:
        """Monitor attention patterns"""
        # Simplified attention monitoring
        attention_pattern = {
            'focus_stability': 0.8,
            'attention_distribution': 0.7,
            'cognitive_load': 0.6
        }
        return attention_pattern

class NeuroSymbolicConsciousnessAgent(nn.Module):
    """
    Main neuro-symbolic consciousness agent integrating all components
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        num_predicates: int = 10,
        num_rules: int = 50
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Core components
        self.ltn_reasoner = LogicTensorNetwork(embedding_dim, num_predicates)
        self.psl_reasoner = ProbabilisticSoftLogic(num_rules, embedding_dim)
        self.meta_cognitive_reasoner = MetaCognitiveReasoner(embedding_dim)
        
        # Integration network
        self.integration_network = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self, 
        query: str, 
        facts: Dict[str, torch.Tensor],
        rules: List[str]
    ) -> Dict[str, Union[torch.Tensor, str, Dict]]:
        """
        Perform comprehensive neuro-symbolic reasoning
        
        Args:
            query: Natural language query
            facts: Known facts
            rules: Logical rules
        
        Returns:
            result: Comprehensive reasoning result
        """
        # LTN reasoning
        ltn_result, ltn_state, ltn_explanation = self.ltn_reasoner(query, facts)
        
        # PSL reasoning
        query_atoms = self._extract_query_atoms(query)
        psl_result, atom_probabilities = self.psl_reasoner(query_atoms, facts, rules)
        
        # Meta-cognitive reflection
        reasoning_process = ltn_state.reasoning_trace
        meta_reflection = self.meta_cognitive_reasoner.reflect_on_reasoning(
            reasoning_process, ltn_result
        )
        
        # Integrate results
        integrated_result = self._integrate_results(ltn_result, psl_result)
        
        # Generate comprehensive explanation
        comprehensive_explanation = self._generate_comprehensive_explanation(
            ltn_explanation, meta_reflection, integrated_result
        )
        
        return {
            'result': integrated_result,
            'ltn_result': ltn_result,
            'psl_result': psl_result,
            'explanation': comprehensive_explanation,
            'meta_reflection': meta_reflection,
            'reasoning_state': ltn_state,
            'atom_probabilities': atom_probabilities
        }
    
    def _extract_query_atoms(self, query: str) -> List[str]:
        """Extract atoms from query"""
        # Simplified atom extraction
        words = query.split()
        atoms = [word for word in words if word.isalpha() and len(word) > 1]
        return atoms
    
    def _integrate_results(self, ltn_result: torch.Tensor, psl_result: torch.Tensor) -> torch.Tensor:
        """Integrate LTN and PSL results"""
        # Combine results
        combined = torch.cat([ltn_result, psl_result.mean().unsqueeze(0), torch.tensor([0.5])])
        
        # Apply integration network
        integrated = self.integration_network(combined)
        
        return integrated
    
    def _generate_comprehensive_explanation(
        self, 
        ltn_explanation: str, 
        meta_reflection: Dict, 
        integrated_result: torch.Tensor
    ) -> str:
        """Generate comprehensive explanation"""
        explanation_parts = [
            f"LTN Reasoning: {ltn_explanation}",
            f"Meta-cognitive Assessment: {meta_reflection.get('self_assessment', 'No assessment available')}",
            f"Confidence: {meta_reflection.get('confidence', 0.5):.2f}",
            f"Final Result: {integrated_result.item():.2f}"
        ]
        
        return " | ".join(explanation_parts)

