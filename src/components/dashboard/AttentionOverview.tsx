'use client';

import React, { useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Network, Eye, Target, Zap, Brain } from 'lucide-react';

interface AttentionFocus {
  id: string;
  element: string;
  intensity: number;
  duration: number;
  timestamp: number;
}

interface MetaCognitiveState {
  selfAwareness: number;
  confidence: number;
  reflection: string;
  adaptation: number;
}

interface AttentionOverviewProps {
  attentionFocus: AttentionFocus[];
  metaCognitiveState: MetaCognitiveState;
}

export function AttentionOverview({ attentionFocus, metaCognitiveState }: AttentionOverviewProps) {
  const attentionStats = useMemo(() => {
    const highIntensity = attentionFocus.filter(focus => focus.intensity >= 0.8).length;
    const mediumIntensity = attentionFocus.filter(focus => focus.intensity >= 0.5 && focus.intensity < 0.8).length;
    const lowIntensity = attentionFocus.filter(focus => focus.intensity < 0.5).length;
    
    const avgIntensity = attentionFocus.length > 0 
      ? attentionFocus.reduce((sum, focus) => sum + focus.intensity, 0) / attentionFocus.length 
      : 0;

    const avgDuration = attentionFocus.length > 0 
      ? attentionFocus.reduce((sum, focus) => sum + focus.duration, 0) / attentionFocus.length 
      : 0;

    const currentFocus = attentionFocus
      .sort((a, b) => b.timestamp - a.timestamp)
      .slice(0, 5);

    const focusElements = attentionFocus.reduce((acc, focus) => {
      acc[focus.element] = (acc[focus.element] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    const mostFocusedElement = Object.entries(focusElements)
      .sort(([,a], [,b]) => b - a)[0];

    return {
      total: attentionFocus.length,
      highIntensity,
      mediumIntensity,
      lowIntensity,
      avgIntensity,
      avgDuration,
      currentFocus,
      mostFocusedElement: mostFocusedElement ? mostFocusedElement[0] : 'None'
    };
  }, [attentionFocus]);

  const getIntensityColor = (intensity: number) => {
    if (intensity >= 0.8) return 'text-red-400';
    if (intensity >= 0.5) return 'text-yellow-400';
    return 'text-green-400';
  };

  const getIntensityIcon = (intensity: number) => {
    if (intensity >= 0.8) return <Target className="w-4 h-4 text-red-400" />;
    if (intensity >= 0.5) return <Eye className="w-4 h-4 text-yellow-400" />;
    return <Eye className="w-4 h-4 text-green-400" />;
  };

  const getDurationText = (duration: number) => {
    if (duration < 1000) return `${duration}ms`;
    if (duration < 60000) return `${(duration / 1000).toFixed(1)}s`;
    return `${(duration / 60000).toFixed(1)}m`;
  };

  return (
    <Card className="attention-card">
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Network className="w-5 h-5 text-attention-400" />
          <span>Attention Overview</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Attention Statistics */}
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400">Total Focus Points</span>
              <span className="text-lg font-bold text-attention-400">{attentionStats.total}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400">Avg Intensity</span>
              <span className={`text-lg font-bold ${getIntensityColor(attentionStats.avgIntensity)}`}>
                {(attentionStats.avgIntensity * 100).toFixed(0)}%
              </span>
            </div>
          </div>
          
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400">Avg Duration</span>
              <span className="text-lg font-bold text-blue-400">
                {getDurationText(attentionStats.avgDuration)}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400">Most Focused</span>
              <span className="text-sm font-medium text-gray-300 truncate max-w-24">
                {attentionStats.mostFocusedElement}
              </span>
            </div>
          </div>
        </div>

        {/* Intensity Distribution */}
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-gray-300">Intensity Distribution</h4>
          
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Target className="w-4 h-4 text-red-400" />
                <span className="text-sm text-gray-400">High (≥80%)</span>
              </div>
              <span className="text-sm font-medium text-red-400">
                {attentionStats.highIntensity}
              </span>
            </div>
            
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Eye className="w-4 h-4 text-yellow-400" />
                <span className="text-sm text-gray-400">Medium (50-79%)</span>
              </div>
              <span className="text-sm font-medium text-yellow-400">
                {attentionStats.mediumIntensity}
              </span>
            </div>
            
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Eye className="w-4 h-4 text-green-400" />
                <span className="text-sm text-gray-400">Low (&lt;50%)</span>
              </div>
              <span className="text-sm font-medium text-green-400">
                {attentionStats.lowIntensity}
              </span>
            </div>
          </div>
        </div>

        {/* Current Focus */}
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-gray-300 flex items-center space-x-2">
            <Zap className="w-4 h-4" />
            <span>Current Focus</span>
          </h4>
          
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {attentionStats.currentFocus.length > 0 ? (
              attentionStats.currentFocus.map((focus) => (
                <div 
                  key={focus.id}
                  className="p-3 bg-gray-800/30 rounded-lg border border-gray-700/50 hover:border-gray-600/50 transition-colors"
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      {getIntensityIcon(focus.intensity)}
                      <span className="text-sm font-medium text-gray-300">
                        {focus.element}
                      </span>
                    </div>
                    <Badge 
                      variant="outline" 
                      className={getIntensityColor(focus.intensity)}
                    >
                      {(focus.intensity * 100).toFixed(0)}%
                    </Badge>
                  </div>
                  
                  <div className="flex items-center justify-between text-xs text-gray-500">
                    <span>Duration: {getDurationText(focus.duration)}</span>
                    <span className={getIntensityColor(focus.intensity)}>
                      {focus.intensity >= 0.8 ? 'Intense' : focus.intensity >= 0.5 ? 'Moderate' : 'Light'}
                    </span>
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center py-8 text-gray-500">
                <Brain className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p className="text-sm">No attention focus yet</p>
              </div>
            )}
          </div>
        </div>

        {/* Attention Quality Indicator */}
        <div className="pt-4 border-t border-gray-700">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-gray-300">Attention Quality</span>
            <div className="flex items-center space-x-2">
              {metaCognitiveState.selfAwareness >= 0.8 ? (
                <>
                  <Target className="w-4 h-4 text-green-400" />
                  <span className="text-sm text-green-400 font-medium">Focused</span>
                </>
              ) : metaCognitiveState.selfAwareness >= 0.6 ? (
                <>
                  <Eye className="w-4 h-4 text-yellow-400" />
                  <span className="text-sm text-yellow-400 font-medium">Moderate</span>
                </>
              ) : (
                <>
                  <Eye className="w-4 h-4 text-red-400" />
                  <span className="text-sm text-red-400 font-medium">Scattered</span>
                </>
              )}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
