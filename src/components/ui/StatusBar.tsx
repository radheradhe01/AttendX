'use client';

import React from 'react';
import { Badge } from '@/components/ui/badge';
import { Brain, Database, Cpu, Network, Clock } from 'lucide-react';

interface ConsciousnessState {
  isActive: boolean;
  currentGoal: string;
  attentionFocus: string[];
  memoryTraces: any[];
  reasoningSteps: any[];
  metaCognitiveState: {
    selfAwareness: number;
    confidence: number;
    reflection: string;
    adaptation: number;
  };
  decisionHistory: any[];
  performanceMetrics: {
    memoryRetention: number;
    reasoningAccuracy: number;
    adaptationSpeed: number;
    selfConsistency: number;
  };
  lastUpdate: number;
}

interface StatusBarProps {
  consciousnessState: ConsciousnessState;
  isAgentActive: boolean;
}

export function StatusBar({ consciousnessState, isAgentActive }: StatusBarProps) {
  const getStatusColor = (value: number) => {
    if (value >= 0.8) return 'text-green-400';
    if (value >= 0.6) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <div className="bg-gray-800/50 backdrop-blur-sm border-b border-gray-700 px-6 py-3">
      <div className="flex items-center justify-between">
        {/* Left side - System Status */}
        <div className="flex items-center space-x-6">
          <div className="flex items-center space-x-2">
            <Brain className="w-4 h-4 text-consciousness-400" />
            <span className="text-sm text-gray-300">Consciousness:</span>
            <Badge 
              variant="outline" 
              className={isAgentActive ? "bg-green-500/20 text-green-400" : "bg-gray-500/20 text-gray-400"}
            >
              {(consciousnessState.metaCognitiveState.selfAwareness * 100).toFixed(0)}%
            </Badge>
          </div>

          <div className="flex items-center space-x-2">
            <Database className="w-4 h-4 text-memory-400" />
            <span className="text-sm text-gray-300">Memory:</span>
            <span className={`text-sm font-medium ${getStatusColor(consciousnessState.performanceMetrics.memoryRetention)}`}>
              {(consciousnessState.performanceMetrics.memoryRetention * 100).toFixed(0)}%
            </span>
          </div>

          <div className="flex items-center space-x-2">
            <Cpu className="w-4 h-4 text-reasoning-400" />
            <span className="text-sm text-gray-300">Reasoning:</span>
            <span className={`text-sm font-medium ${getStatusColor(consciousnessState.performanceMetrics.reasoningAccuracy)}`}>
              {(consciousnessState.performanceMetrics.reasoningAccuracy * 100).toFixed(0)}%
            </span>
          </div>

          <div className="flex items-center space-x-2">
            <Network className="w-4 h-4 text-attention-400" />
            <span className="text-sm text-gray-300">Attention:</span>
            <span className="text-sm font-medium text-attention-400">
              {consciousnessState.attentionFocus.length} foci
            </span>
          </div>
        </div>

        {/* Right side - Timestamps and Counts */}
        <div className="flex items-center space-x-6 text-sm text-gray-400">
          <div className="flex items-center space-x-2">
            <Clock className="w-4 h-4" />
            <span>Last update: {new Date(consciousnessState.lastUpdate).toLocaleTimeString()}</span>
          </div>

          <div className="flex items-center space-x-4">
            <span>Traces: {consciousnessState.memoryTraces.length}</span>
            <span>Steps: {consciousnessState.reasoningSteps.length}</span>
            <span>Decisions: {consciousnessState.decisionHistory.length}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
