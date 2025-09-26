'use client';

import React from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Play, Square, RotateCcw, Brain, Zap } from 'lucide-react';

interface HeaderProps {
  isAgentActive: boolean;
  onToggleAgent: () => void;
  onResetAgent: () => void;
}

export function Header({ isAgentActive, onToggleAgent, onResetAgent }: HeaderProps) {
  return (
    <header className="bg-gray-900/95 backdrop-blur-sm border-b border-gray-700 px-6 py-4">
      <div className="flex items-center justify-between">
        {/* Left side - Title and Status */}
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <Brain className="w-6 h-6 text-consciousness-400" />
            <h1 className="text-xl font-bold text-white">Artificial Consciousness Simulator</h1>
          </div>
          
          <Badge 
            variant="outline" 
            className={isAgentActive ? "bg-green-500/20 text-green-400 border-green-500/30" : "bg-gray-500/20 text-gray-400 border-gray-500/30"}
          >
            <div className={`w-2 h-2 rounded-full mr-2 ${isAgentActive ? 'bg-green-400' : 'bg-gray-400'}`} />
            {isAgentActive ? 'Active' : 'Inactive'}
          </Badge>
        </div>

        {/* Right side - Controls */}
        <div className="flex items-center space-x-3">
          <Button
            onClick={onToggleAgent}
            variant={isAgentActive ? "destructive" : "default"}
            size="sm"
            className="flex items-center space-x-2"
          >
            {isAgentActive ? (
              <>
                <Square className="w-4 h-4" />
                <span>Stop Agent</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                <span>Start Agent</span>
              </>
            )}
          </Button>
          
          <Button
            onClick={onResetAgent}
            variant="outline"
            size="sm"
            className="flex items-center space-x-2"
          >
            <RotateCcw className="w-4 h-4" />
            <span>Reset</span>
          </Button>
        </div>
      </div>
    </header>
  );
}
