'use client';

import React from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Play, Square, RotateCcw, Settings, Brain, Zap } from 'lucide-react';

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

interface AgentControlsProps {
  isAgentActive: boolean;
  onStart: () => void;
  onStop: () => void;
  onReset: () => void;
  consciousnessState: ConsciousnessState;
}

export function AgentControls({ 
  isAgentActive, 
  onStart, 
  onStop, 
  onReset, 
  consciousnessState 
}: AgentControlsProps) {
  return (
    <div className="px-6 py-4">
      <div className="flex items-center justify-between">
        {/* Left side - Agent Status */}
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <Brain className="w-5 h-5 text-consciousness-400" />
            <span className="text-sm font-medium text-gray-300">Agent Status:</span>
            <Badge 
              variant="outline" 
              className={isAgentActive ? "bg-green-500/20 text-green-400 border-green-500/30" : "bg-gray-500/20 text-gray-400 border-gray-500/30"}
            >
              <div className={`w-2 h-2 rounded-full mr-2 ${isAgentActive ? 'bg-green-400 animate-pulse' : 'bg-gray-400'}`} />
              {isAgentActive ? 'Conscious' : 'Dormant'}
            </Badge>
          </div>

          <div className="flex items-center space-x-2">
            <Zap className="w-4 h-4 text-yellow-400" />
            <span className="text-sm text-gray-400">Confidence:</span>
            <span className="text-sm font-medium text-yellow-400">
              {(consciousnessState.metaCognitiveState.confidence * 100).toFixed(0)}%
            </span>
          </div>
        </div>

        {/* Right side - Control Buttons */}
        <div className="flex items-center space-x-3">
          {!isAgentActive ? (
            <Button
              onClick={onStart}
              className="flex items-center space-x-2 bg-green-600 hover:bg-green-700 text-white"
            >
              <Play className="w-4 h-4" />
              <span>Activate Agent</span>
            </Button>
          ) : (
            <Button
              onClick={onStop}
              variant="destructive"
              className="flex items-center space-x-2"
            >
              <Square className="w-4 h-4" />
              <span>Deactivate Agent</span>
            </Button>
          )}

          <Button
            onClick={onReset}
            variant="outline"
            className="flex items-center space-x-2"
          >
            <RotateCcw className="w-4 h-4" />
            <span>Reset</span>
          </Button>

          <Button
            variant="ghost"
            size="sm"
            className="flex items-center space-x-2"
          >
            <Settings className="w-4 h-4" />
            <span>Settings</span>
          </Button>
        </div>
      </div>

      {/* Current Goal Display */}
      <div className="mt-4 p-3 bg-gray-800/30 rounded-lg border border-gray-700/50">
        <div className="flex items-center space-x-2 mb-2">
          <span className="text-sm font-medium text-gray-300">Current Goal:</span>
        </div>
        <p className="text-sm text-gray-400">{consciousnessState.currentGoal}</p>
      </div>
    </div>
  );
}
