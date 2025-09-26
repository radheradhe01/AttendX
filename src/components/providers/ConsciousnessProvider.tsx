'use client';

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { io, Socket } from 'socket.io-client';

export interface ConsciousnessState {
  isActive: boolean;
  currentGoal: string;
  attentionFocus: AttentionFocus[];
  memoryTraces: MemoryTrace[];
  reasoningSteps: ReasoningStep[];
  metaCognitiveState: MetaCognitiveState;
  decisionHistory: Decision[];
  performanceMetrics: PerformanceMetrics;
  lastUpdate: number;
}

export interface MemoryTrace {
  id: string;
  type: 'episodic' | 'semantic';
  content: string;
  timestamp: number;
  relevance: number;
  attention: number;
}

export interface ReasoningStep {
  id: string;
  rule: string;
  confidence: number;
  timestamp: number;
  explanation: string;
}

export interface MetaCognitiveState {
  selfAwareness: number;
  confidence: number;
  reflection: string;
  adaptation: number;
}

export interface AttentionFocus {
  id: string;
  element: string;
  intensity: number;
  duration: number;
  timestamp: number;
}

export interface Decision {
  id: string;
  action: string;
  horizon: 'short' | 'medium' | 'long';
  confidence: number;
  reasoning: string;
  timestamp: number;
  outcome?: string;
}

export interface PerformanceMetrics {
  memoryRetention: number;
  reasoningAccuracy: number;
  adaptationSpeed: number;
  selfConsistency: number;
}

export interface ConsciousnessContextType {
  consciousnessState: ConsciousnessState;
  startAgent: () => void;
  stopAgent: () => void;
  resetAgent: () => void;
  updateGoal: (goal: string) => void;
  addMemoryTrace: (trace: Omit<MemoryTrace, 'id' | 'timestamp'>) => void;
  addReasoningStep: (step: Omit<ReasoningStep, 'id' | 'timestamp'>) => void;
  addDecision: (decision: Omit<Decision, 'id' | 'timestamp'>) => void;
  socket: Socket | null;
  isConnected: boolean;
}

export const ConsciousnessContext = createContext<ConsciousnessContextType | undefined>(undefined);

const initialState: ConsciousnessState = {
  isActive: false,
  currentGoal: 'Explore and learn about the environment',
  attentionFocus: [],
  memoryTraces: [],
  reasoningSteps: [],
  metaCognitiveState: {
    selfAwareness: 0.5,
    confidence: 0.5,
    reflection: 'Initializing consciousness...',
    adaptation: 0.5,
  },
  decisionHistory: [],
  performanceMetrics: {
    memoryRetention: 0.8,
    reasoningAccuracy: 0.7,
    adaptationSpeed: 0.6,
    selfConsistency: 0.75,
  },
  lastUpdate: Date.now(),
};

export function ConsciousnessProvider({ children }: { children: ReactNode }) {
  const [consciousnessState, setConsciousnessState] = useState<ConsciousnessState>(initialState);
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    // Initialize WebSocket connection
    const newSocket = io(process.env.NEXT_PUBLIC_WEBSOCKET_URL || 'ws://localhost:8000', {
      transports: ['websocket'],
      autoConnect: false,
    });

    newSocket.on('connect', () => {
      console.log('Connected to consciousness agent');
      setIsConnected(true);
    });

    newSocket.on('disconnect', () => {
      console.log('Disconnected from consciousness agent');
      setIsConnected(false);
    });

    newSocket.on('consciousness_update', (data: Partial<ConsciousnessState>) => {
      setConsciousnessState(prev => ({
        ...prev,
        ...data,
        lastUpdate: Date.now(),
      }));
    });

    newSocket.on('memory_trace', (trace: MemoryTrace) => {
      setConsciousnessState(prev => ({
        ...prev,
        memoryTraces: [...prev.memoryTraces.slice(-99), trace], // Keep last 100 traces
        lastUpdate: Date.now(),
      }));
    });

    newSocket.on('reasoning_step', (step: ReasoningStep) => {
      setConsciousnessState(prev => ({
        ...prev,
        reasoningSteps: [...prev.reasoningSteps.slice(-49), step], // Keep last 50 steps
        lastUpdate: Date.now(),
      }));
    });

    newSocket.on('decision', (decision: Decision) => {
      setConsciousnessState(prev => ({
        ...prev,
        decisionHistory: [...prev.decisionHistory.slice(-99), decision], // Keep last 100 decisions
        lastUpdate: Date.now(),
      }));
    });

    newSocket.on('meta_cognitive_update', (metaState: MetaCognitiveState) => {
      setConsciousnessState(prev => ({
        ...prev,
        metaCognitiveState: metaState,
        lastUpdate: Date.now(),
      }));
    });

    setSocket(newSocket);

    return () => {
      newSocket.close();
    };
  }, []);

  const startAgent = () => {
    if (socket) {
      socket.connect();
      socket.emit('start_agent');
      setConsciousnessState(prev => ({
        ...prev,
        isActive: true,
        lastUpdate: Date.now(),
      }));
    }
  };

  const stopAgent = () => {
    if (socket) {
      socket.emit('stop_agent');
      setConsciousnessState(prev => ({
        ...prev,
        isActive: false,
        lastUpdate: Date.now(),
      }));
    }
  };

  const resetAgent = () => {
    if (socket) {
      socket.emit('reset_agent');
      setConsciousnessState(initialState);
    }
  };

  const updateGoal = (goal: string) => {
    if (socket) {
      socket.emit('update_goal', goal);
      setConsciousnessState(prev => ({
        ...prev,
        currentGoal: goal,
        lastUpdate: Date.now(),
      }));
    }
  };

  const addMemoryTrace = (trace: Omit<MemoryTrace, 'id' | 'timestamp'>) => {
    const newTrace: MemoryTrace = {
      ...trace,
      id: `trace_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: Date.now(),
    };

    setConsciousnessState(prev => ({
      ...prev,
      memoryTraces: [...prev.memoryTraces.slice(-99), newTrace],
      lastUpdate: Date.now(),
    }));
  };

  const addReasoningStep = (step: Omit<ReasoningStep, 'id' | 'timestamp'>) => {
    const newStep: ReasoningStep = {
      ...step,
      id: `step_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: Date.now(),
    };

    setConsciousnessState(prev => ({
      ...prev,
      reasoningSteps: [...prev.reasoningSteps.slice(-49), newStep],
      lastUpdate: Date.now(),
    }));
  };

  const addDecision = (decision: Omit<Decision, 'id' | 'timestamp'>) => {
    const newDecision: Decision = {
      ...decision,
      id: `decision_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: Date.now(),
    };

    setConsciousnessState(prev => ({
      ...prev,
      decisionHistory: [...prev.decisionHistory.slice(-99), newDecision],
      lastUpdate: Date.now(),
    }));
  };

  const value: ConsciousnessContextType = {
    consciousnessState,
    startAgent,
    stopAgent,
    resetAgent,
    updateGoal,
    addMemoryTrace,
    addReasoningStep,
    addDecision,
    socket,
    isConnected,
  };

  return (
    <ConsciousnessContext.Provider value={value}>
      {children}
    </ConsciousnessContext.Provider>
  );
}

// The consumer hook lives in `src/hooks/useConsciousness.ts`.
