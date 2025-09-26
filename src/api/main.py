"""
FastAPI Backend for Artificial Consciousness Simulator
Orchestrates the proto-conscious agent and provides real-time data streaming
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Import our consciousness components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

from dnc_memory import DNCMemorySystem, EpisodicMemorySystem, SemanticMemorySystem
from meta_learning import MAMLAgent, ContinualLearningAgent
from neuro_symbolic import NeuroSymbolicConsciousnessAgent
from rl_orchestration import RLOrchestrationSystem, MultiHorizonRLAgent

app = FastAPI(
    title="Artificial Consciousness Simulator API",
    description="Real-time proto-conscious agent orchestration and monitoring",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance
consciousness_agent = None
active_connections: List[WebSocket] = []

# Pydantic models
class AgentState(BaseModel):
    is_active: bool
    current_goal: str
    consciousness_level: float
    confidence: float
    memory_traces_count: int
    reasoning_steps_count: int
    decision_count: int
    last_update: float

class MemoryTrace(BaseModel):
    id: str
    type: str  # 'episodic' or 'semantic'
    content: str
    timestamp: float
    relevance: float
    attention: float

class ReasoningStep(BaseModel):
    id: str
    rule: str
    confidence: float
    timestamp: float
    explanation: str

class Decision(BaseModel):
    id: str
    action: str
    horizon: str  # 'short', 'medium', 'long'
    confidence: float
    reasoning: str
    timestamp: float
    outcome: Optional[str] = None

class MetaCognitiveState(BaseModel):
    self_awareness: float
    confidence: float
    reflection: str
    adaptation: float

class PerformanceMetrics(BaseModel):
    memory_retention: float
    reasoning_accuracy: float
    adaptation_speed: float
    self_consistency: float

class ConsciousnessUpdate(BaseModel):
    agent_state: AgentState
    memory_traces: List[MemoryTrace]
    reasoning_steps: List[ReasoningStep]
    decisions: List[Decision]
    meta_cognitive_state: MetaCognitiveState
    performance_metrics: PerformanceMetrics
    attention_focus: List[str]

class ConsciousnessAgent:
    """
    Main consciousness agent orchestrating all components
    """
    
    def __init__(self):
        self.is_active = False
        self.current_goal = "Explore and learn about the environment"
        
        # Initialize core components
        self.memory_system = DNCMemorySystem()
        self.episodic_memory = EpisodicMemorySystem()
        self.semantic_memory = SemanticMemorySystem()
        
        self.meta_learning_agent = ContinualLearningAgent(
            input_dim=64,
            hidden_dim=128,
            output_dim=32
        )
        
        self.neuro_symbolic_reasoner = NeuroSymbolicConsciousnessAgent()
        
        self.rl_system = RLOrchestrationSystem({
            'short_term_config': {},
            'medium_term_config': {},
            'long_term_config': {},
            'meta_config': {},
            'env_config': {}
        })
        
        # State tracking
        self.memory_traces = []
        self.reasoning_steps = []
        self.decisions = []
        self.attention_focus = []
        
        # Performance metrics
        self.performance_metrics = PerformanceMetrics(
            memory_retention=0.8,
            reasoning_accuracy=0.7,
            adaptation_speed=0.6,
            self_consistency=0.75
        )
        
        # Meta-cognitive state
        self.meta_cognitive_state = MetaCognitiveState(
            self_awareness=0.5,
            confidence=0.5,
            reflection="Initializing consciousness...",
            adaptation=0.5
        )
        
        # Consciousness simulation task
        self.consciousness_task = None
        
    async def start(self):
        """Start the consciousness agent"""
        if self.is_active:
            return
        
        self.is_active = True
        self.meta_cognitive_state.reflection = "Consciousness activated. Beginning self-awareness..."
        
        # Start consciousness simulation loop
        self.consciousness_task = asyncio.create_task(self._consciousness_loop())
        
        await self._broadcast_update()
        
    async def stop(self):
        """Stop the consciousness agent"""
        if not self.is_active:
            return
        
        self.is_active = False
        self.meta_cognitive_state.reflection = "Consciousness deactivated. Entering dormant state..."
        
        if self.consciousness_task:
            self.consciousness_task.cancel()
            try:
                await self.consciousness_task
            except asyncio.CancelledError:
                pass
        
        await self._broadcast_update()
        
    async def reset(self):
        """Reset the consciousness agent"""
        await self.stop()
        
        # Reset all state
        self.memory_traces = []
        self.reasoning_steps = []
        self.decisions = []
        self.attention_focus = []
        
        self.meta_cognitive_state = MetaCognitiveState(
            self_awareness=0.5,
            confidence=0.5,
            reflection="Consciousness reset. Ready for new experiences...",
            adaptation=0.5
        )
        
        self.performance_metrics = PerformanceMetrics(
            memory_retention=0.8,
            reasoning_accuracy=0.7,
            adaptation_speed=0.6,
            self_consistency=0.75
        )
        
        await self._broadcast_update()
        
    async def update_goal(self, new_goal: str):
        """Update the current goal"""
        self.current_goal = new_goal
        self.meta_cognitive_state.reflection = f"Goal updated: {new_goal}"
        await self._broadcast_update()
        
    async def _consciousness_loop(self):
        """Main consciousness simulation loop"""
        while self.is_active:
            try:
                # Simulate consciousness processes
                await self._simulate_consciousness_cycle()
                
                # Update meta-cognitive state
                await self._update_meta_cognitive_state()
                
                # Broadcast updates
                await self._broadcast_update()
                
                # Wait before next cycle
                await asyncio.sleep(2.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in consciousness loop: {e}")
                await asyncio.sleep(1.0)
                
    async def _simulate_consciousness_cycle(self):
        """Simulate one cycle of consciousness processes"""
        current_time = time.time()
        
        # Simulate memory formation
        if len(self.memory_traces) < 100:  # Limit memory traces
            memory_trace = MemoryTrace(
                id=str(uuid.uuid4()),
                type='episodic' if len(self.memory_traces) % 2 == 0 else 'semantic',
                content=f"Memory trace {len(self.memory_traces) + 1}: Processing environmental input",
                timestamp=current_time,
                relevance=0.7 + (hash(str(current_time)) % 30) / 100,
                attention=0.6 + (hash(str(current_time)) % 40) / 100
            )
            self.memory_traces.append(memory_trace)
            
        # Simulate reasoning steps
        if len(self.reasoning_steps) < 50:  # Limit reasoning steps
            reasoning_step = ReasoningStep(
                id=str(uuid.uuid4()),
                rule=f"Rule_{len(self.reasoning_steps) + 1}",
                confidence=0.6 + (hash(str(current_time)) % 40) / 100,
                timestamp=current_time,
                explanation=f"Applied reasoning rule {len(self.reasoning_steps) + 1} to current situation"
            )
            self.reasoning_steps.append(reasoning_step)
            
        # Simulate decisions
        if len(self.decisions) < 100:  # Limit decisions
            horizon = ['short', 'medium', 'long'][len(self.decisions) % 3]
            decision = Decision(
                id=str(uuid.uuid4()),
                action=f"Action_{len(self.decisions) + 1}",
                horizon=horizon,
                confidence=0.5 + (hash(str(current_time)) % 50) / 100,
                reasoning=f"Decision based on {horizon}-term planning",
                timestamp=current_time,
                outcome='success' if len(self.decisions) % 4 != 0 else 'failure'
            )
            self.decisions.append(decision)
            
        # Simulate attention focus
        attention_elements = [
            "environmental_sensors", "goal_progress", "memory_retrieval",
            "reasoning_process", "decision_making", "self_monitoring"
        ]
        self.attention_focus = attention_elements[:3 + (len(self.decisions) % 3)]
        
    async def _update_meta_cognitive_state(self):
        """Update meta-cognitive state based on current performance"""
        # Simulate gradual improvement in self-awareness
        if self.meta_cognitive_state.self_awareness < 0.9:
            self.meta_cognitive_state.self_awareness += 0.001
            
        # Update confidence based on recent performance
        recent_decisions = self.decisions[-10:] if len(self.decisions) >= 10 else self.decisions
        if recent_decisions:
            success_rate = sum(1 for d in recent_decisions if d.outcome == 'success') / len(recent_decisions)
            self.meta_cognitive_state.confidence = 0.3 + (success_rate * 0.7)
            
        # Update reflection
        reflection_templates = [
            "Monitoring current goal progress...",
            "Reflecting on recent decision outcomes...",
            "Adapting to new environmental information...",
            "Consolidating episodic memories...",
            "Evaluating reasoning consistency...",
            "Assessing attention focus distribution...",
            "Updating self-model based on experiences...",
            "Planning next action sequence...",
            "Analyzing meta-cognitive performance...",
            "Integrating multi-modal information..."
        ]
        
        current_reflection = reflection_templates[len(self.decisions) % len(reflection_templates)]
        self.meta_cognitive_state.reflection = current_reflection
        
        # Update adaptation capability
        self.meta_cognitive_state.adaptation = min(0.9, 0.5 + (len(self.memory_traces) * 0.001))
        
    async def _broadcast_update(self):
        """Broadcast consciousness update to all connected clients"""
        if not active_connections:
            return
            
        # Prepare update data
        update_data = ConsciousnessUpdate(
            agent_state=AgentState(
                is_active=self.is_active,
                current_goal=self.current_goal,
                consciousness_level=self.meta_cognitive_state.self_awareness,
                confidence=self.meta_cognitive_state.confidence,
                memory_traces_count=len(self.memory_traces),
                reasoning_steps_count=len(self.reasoning_steps),
                decision_count=len(self.decisions),
                last_update=time.time()
            ),
            memory_traces=self.memory_traces[-10:],  # Send last 10
            reasoning_steps=self.reasoning_steps[-10:],  # Send last 10
            decisions=self.decisions[-10:],  # Send last 10
            meta_cognitive_state=self.meta_cognitive_state,
            performance_metrics=self.performance_metrics,
            attention_focus=self.attention_focus
        )
        
        # Broadcast to all connections
        message = {
            "type": "consciousness_update",
            "data": update_data.dict()
        }
        
        disconnected = []
        for connection in active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                disconnected.append(connection)
                
        # Remove disconnected connections
        for connection in disconnected:
            active_connections.remove(connection)

# Initialize global agent
consciousness_agent = ConsciousnessAgent()

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Artificial Consciousness Simulator API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/agent/status")
async def get_agent_status():
    """Get current agent status"""
    return {
        "is_active": consciousness_agent.is_active,
        "current_goal": consciousness_agent.current_goal,
        "consciousness_level": consciousness_agent.meta_cognitive_state.self_awareness,
        "confidence": consciousness_agent.meta_cognitive_state.confidence,
        "memory_traces_count": len(consciousness_agent.memory_traces),
        "reasoning_steps_count": len(consciousness_agent.reasoning_steps),
        "decision_count": len(consciousness_agent.decisions),
        "last_update": time.time()
    }

@app.post("/agent/start")
async def start_agent():
    """Start the consciousness agent"""
    await consciousness_agent.start()
    return {"message": "Agent started", "status": "active"}

@app.post("/agent/stop")
async def stop_agent():
    """Stop the consciousness agent"""
    await consciousness_agent.stop()
    return {"message": "Agent stopped", "status": "inactive"}

@app.post("/agent/reset")
async def reset_agent():
    """Reset the consciousness agent"""
    await consciousness_agent.reset()
    return {"message": "Agent reset", "status": "reset"}

@app.post("/agent/goal")
async def update_goal(goal_data: dict):
    """Update agent goal"""
    new_goal = goal_data.get("goal", "")
    if not new_goal:
        raise HTTPException(status_code=400, detail="Goal is required")
    
    await consciousness_agent.update_goal(new_goal)
    return {"message": f"Goal updated to: {new_goal}"}

@app.get("/agent/memory")
async def get_memory_traces():
    """Get memory traces"""
    return {
        "traces": consciousness_agent.memory_traces[-20:],  # Last 20 traces
        "total_count": len(consciousness_agent.memory_traces)
    }

@app.get("/agent/reasoning")
async def get_reasoning_steps():
    """Get reasoning steps"""
    return {
        "steps": consciousness_agent.reasoning_steps[-20:],  # Last 20 steps
        "total_count": len(consciousness_agent.reasoning_steps)
    }

@app.get("/agent/decisions")
async def get_decisions():
    """Get decisions"""
    return {
        "decisions": consciousness_agent.decisions[-20:],  # Last 20 decisions
        "total_count": len(consciousness_agent.decisions)
    }

@app.get("/agent/metrics")
async def get_performance_metrics():
    """Get performance metrics"""
    return consciousness_agent.performance_metrics.dict()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time consciousness updates"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Keep connection alive and handle client messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            elif message.get("type") == "start_agent":
                await consciousness_agent.start()
            elif message.get("type") == "stop_agent":
                await consciousness_agent.stop()
            elif message.get("type") == "reset_agent":
                await consciousness_agent.reset()
            elif message.get("type") == "update_goal":
                goal = message.get("goal", "")
                if goal:
                    await consciousness_agent.update_goal(goal)
                    
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
