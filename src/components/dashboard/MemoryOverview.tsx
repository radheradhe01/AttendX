'use client';

import React, { useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Database, Clock, TrendingUp, Brain } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';

interface MemoryTrace {
  id: string;
  type: 'episodic' | 'semantic';
  content: string;
  timestamp: number;
  relevance: number;
  attention: number;
}

interface PerformanceMetrics {
  memoryRetention: number;
  reasoningAccuracy: number;
  adaptationSpeed: number;
  selfConsistency: number;
}

interface MemoryOverviewProps {
  memoryTraces: MemoryTrace[];
  performanceMetrics: PerformanceMetrics;
}

export function MemoryOverview({ memoryTraces, performanceMetrics }: MemoryOverviewProps) {
  const memoryStats = useMemo(() => {
    const episodic = memoryTraces.filter(trace => trace.type === 'episodic');
    const semantic = memoryTraces.filter(trace => trace.type === 'semantic');
    
    const avgRelevance = memoryTraces.length > 0 
      ? memoryTraces.reduce((sum, trace) => sum + trace.relevance, 0) / memoryTraces.length 
      : 0;
    
    const avgAttention = memoryTraces.length > 0 
      ? memoryTraces.reduce((sum, trace) => sum + trace.attention, 0) / memoryTraces.length 
      : 0;

    const recentTraces = memoryTraces
      .sort((a, b) => b.timestamp - a.timestamp)
      .slice(0, 5);

    return {
      episodic: episodic.length,
      semantic: semantic.length,
      total: memoryTraces.length,
      avgRelevance,
      avgAttention,
      recentTraces
    };
  }, [memoryTraces]);

  const getTypeColor = (type: 'episodic' | 'semantic') => {
    return type === 'episodic' ? 'bg-blue-500/20 text-blue-400' : 'bg-purple-500/20 text-purple-400';
  };

  const getRelevanceColor = (relevance: number) => {
    if (relevance >= 0.8) return 'text-green-400';
    if (relevance >= 0.6) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <Card className="memory-card">
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Database className="w-5 h-5 text-memory-400" />
          <span>Memory Overview</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Memory Statistics */}
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400">Episodic</span>
              <span className="text-lg font-bold text-blue-400">{memoryStats.episodic}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400">Semantic</span>
              <span className="text-lg font-bold text-purple-400">{memoryStats.semantic}</span>
            </div>
          </div>
          
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400">Total Traces</span>
              <span className="text-lg font-bold text-memory-400">{memoryStats.total}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400">Retention</span>
              <span className="text-lg font-bold text-green-400">
                {(performanceMetrics.memoryRetention * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        </div>

        {/* Memory Quality Metrics */}
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-gray-300">Memory Quality</h4>
          
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400">Avg Relevance</span>
              <span className={`text-sm font-medium ${getRelevanceColor(memoryStats.avgRelevance)}`}>
                {(memoryStats.avgRelevance * 100).toFixed(1)}%
              </span>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400">Avg Attention</span>
              <span className={`text-sm font-medium ${getRelevanceColor(memoryStats.avgAttention)}`}>
                {(memoryStats.avgAttention * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>

        {/* Recent Memory Traces */}
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-gray-300 flex items-center space-x-2">
            <Clock className="w-4 h-4" />
            <span>Recent Traces</span>
          </h4>
          
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {memoryStats.recentTraces.length > 0 ? (
              memoryStats.recentTraces.map((trace) => (
                <div 
                  key={trace.id}
                  className="p-3 bg-gray-800/30 rounded-lg border border-gray-700/50 hover:border-gray-600/50 transition-colors"
                >
                  <div className="flex items-start justify-between mb-2">
                    <Badge className={getTypeColor(trace.type)}>
                      {trace.type}
                    </Badge>
                    <span className="text-xs text-gray-500">
                      {formatDistanceToNow(new Date(trace.timestamp), { addSuffix: true })}
                    </span>
                  </div>
                  
                  <p className="text-sm text-gray-300 mb-2 line-clamp-2">
                    {trace.content}
                  </p>
                  
                  <div className="flex items-center justify-between text-xs text-gray-500">
                    <span>Relevance: {(trace.relevance * 100).toFixed(0)}%</span>
                    <span>Attention: {(trace.attention * 100).toFixed(0)}%</span>
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center py-8 text-gray-500">
                <Brain className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p className="text-sm">No memory traces yet</p>
              </div>
            )}
          </div>
        </div>

        {/* Memory Health Indicator */}
        <div className="pt-4 border-t border-gray-700">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-gray-300">Memory Health</span>
            <div className="flex items-center space-x-2">
              <TrendingUp className="w-4 h-4 text-green-400" />
              <span className="text-sm text-green-400 font-medium">Healthy</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
