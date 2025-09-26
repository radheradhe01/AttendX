'use client';

import React, { useEffect, useRef, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Zap, Brain, Clock, Eye } from 'lucide-react';
import { useConsciousness } from '@/hooks/useConsciousness';
import { formatDistanceToNow } from 'date-fns';

export function StreamOfConsciousness() {
  const { consciousnessState } = useConsciousness();
  const [streamEntries, setStreamEntries] = useState<any[]>([]);
  const streamRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Simulate stream of consciousness entries
    const interval = setInterval(() => {
      if (consciousnessState.isActive) {
        const newEntry = {
          id: `stream_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          timestamp: Date.now(),
          type: ['thought', 'memory', 'reasoning', 'attention'][Math.floor(Math.random() * 4)],
          content: generateStreamContent(),
          intensity: Math.random(),
          confidence: Math.random()
        };
        
        setStreamEntries(prev => [newEntry, ...prev.slice(0, 49)]); // Keep last 50 entries
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [consciousnessState.isActive]);

  useEffect(() => {
    // Auto-scroll to top when new entries are added
    if (streamRef.current) {
      streamRef.current.scrollTop = 0;
    }
  }, [streamEntries]);

  const generateStreamContent = () => {
    const thoughts = [
      "Processing current environment...",
      "Retrieving relevant memories...",
      "Evaluating decision options...",
      "Updating self-model...",
      "Monitoring attention focus...",
      "Reflecting on recent actions...",
      "Adapting to new information...",
      "Consolidating experiences...",
      "Planning next steps...",
      "Assessing goal progress..."
    ];
    return thoughts[Math.floor(Math.random() * thoughts.length)];
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'thought': return 'bg-blue-500/20 text-blue-400';
      case 'memory': return 'bg-purple-500/20 text-purple-400';
      case 'reasoning': return 'bg-green-500/20 text-green-400';
      case 'attention': return 'bg-yellow-500/20 text-yellow-400';
      default: return 'bg-gray-500/20 text-gray-400';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'thought': return Brain;
      case 'memory': return Clock;
      case 'reasoning': return Zap;
      case 'attention': return Eye;
      default: return Brain;
    }
  };

  return (
    <div className="h-full flex flex-col">
      <Card className="flex-1 consciousness-card">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Zap className="w-5 h-5 text-consciousness-400" />
            <span>Stream of Consciousness</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="flex-1 flex flex-col">
          {/* Stream Status */}
          <div className="mb-6 p-4 bg-gray-800/50 rounded-lg border border-gray-700">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2">
                  <div className={`w-3 h-3 rounded-full ${consciousnessState.isActive ? 'bg-green-400 animate-pulse' : 'bg-gray-400'}`} />
                  <span className="text-sm font-medium text-gray-300">
                    {consciousnessState.isActive ? 'Stream Active' : 'Stream Dormant'}
                  </span>
                </div>
                <Badge variant="outline" className="text-xs">
                  {streamEntries.length} entries
                </Badge>
              </div>
              <div className="text-sm text-gray-400">
                Consciousness Level: {(consciousnessState.metaCognitiveState.selfAwareness * 100).toFixed(0)}%
              </div>
            </div>
          </div>

          {/* Stream Entries */}
          <div 
            ref={streamRef}
            className="flex-1 overflow-y-auto space-y-3 max-h-96"
          >
            {streamEntries.length > 0 ? (
              streamEntries.map((entry) => {
                const Icon = getTypeIcon(entry.type);
                return (
                  <div 
                    key={entry.id}
                    className="p-4 bg-gray-800/30 rounded-lg border border-gray-700/50 hover:border-gray-600/50 transition-all duration-300"
                  >
                    <div className="flex items-start space-x-3">
                      {/* Entry Icon */}
                      <div className="flex-shrink-0">
                        <div className={`p-2 rounded-lg ${getTypeColor(entry.type)}`}>
                          <Icon className="w-4 h-4" />
                        </div>
                      </div>

                      {/* Entry Content */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center space-x-2">
                            <Badge className={getTypeColor(entry.type)}>
                              {entry.type}
                            </Badge>
                            <span className="text-xs text-gray-500">
                              {formatDistanceToNow(new Date(entry.timestamp), { addSuffix: true })}
                            </span>
                          </div>
                          <div className="flex items-center space-x-2 text-xs text-gray-500">
                            <span>Intensity: {(entry.intensity * 100).toFixed(0)}%</span>
                            <span>Confidence: {(entry.confidence * 100).toFixed(0)}%</span>
                          </div>
                        </div>
                        
                        <p className="text-sm text-gray-300 leading-relaxed">
                          {entry.content}
                        </p>
                      </div>
                    </div>
                  </div>
                );
              })
            ) : (
              <div className="text-center py-12 text-gray-500">
                <Brain className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p className="text-lg font-medium mb-2">No consciousness stream</p>
                <p className="text-sm">Activate the agent to begin the stream of consciousness</p>
              </div>
            )}
          </div>

          {/* Stream Controls */}
          <div className="mt-6 pt-4 border-t border-gray-700">
            <div className="flex items-center justify-between">
              <div className="text-sm text-gray-400">
                Real-time consciousness monitoring
              </div>
              <div className="flex items-center space-x-2 text-xs text-gray-500">
                <span>Update frequency: 2s</span>
                <span>•</span>
                <span>Buffer size: 50 entries</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}