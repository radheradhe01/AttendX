'use client';

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { TrendingUp, TrendingDown, Minus, Brain, Zap, Target, CheckCircle } from 'lucide-react';

interface PerformanceMetrics {
  memoryRetention: number;
  reasoningAccuracy: number;
  adaptationSpeed: number;
  selfConsistency: number;
}

interface MetaCognitiveState {
  selfAwareness: number;
  confidence: number;
  reflection: string;
  adaptation: number;
}

interface ConsciousnessMetricsProps {
  metrics: PerformanceMetrics;
  metaCognitiveState: MetaCognitiveState;
}

export function ConsciousnessMetrics({ metrics, metaCognitiveState }: ConsciousnessMetricsProps) {
  const getTrendIcon = (value: number, threshold: number = 0.7) => {
    if (value > threshold) return <TrendingUp className="w-4 h-4 text-green-400" />;
    if (value < threshold - 0.2) return <TrendingDown className="w-4 h-4 text-red-400" />;
    return <Minus className="w-4 h-4 text-yellow-400" />;
  };

  const getMetricColor = (value: number) => {
    if (value >= 0.8) return 'text-green-400';
    if (value >= 0.6) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getProgressColor = (value: number) => {
    if (value >= 0.8) return 'bg-green-500';
    if (value >= 0.6) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <Card className="consciousness-card">
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Brain className="w-5 h-5 text-consciousness-400" />
          <span>Consciousness Metrics</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Performance Metrics */}
        <div className="space-y-4">
          <h4 className="text-sm font-medium text-gray-300 flex items-center space-x-2">
            <Target className="w-4 h-4" />
            <span>Performance Metrics</span>
          </h4>
          
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-400">Memory Retention</span>
                <div className="flex items-center space-x-1">
                  {getTrendIcon(metrics.memoryRetention)}
                  <span className={`text-sm font-medium ${getMetricColor(metrics.memoryRetention)}`}>
                    {(metrics.memoryRetention * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
              <Progress 
                value={metrics.memoryRetention * 100} 
                className="h-2"
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-400">Reasoning Accuracy</span>
                <div className="flex items-center space-x-1">
                  {getTrendIcon(metrics.reasoningAccuracy)}
                  <span className={`text-sm font-medium ${getMetricColor(metrics.reasoningAccuracy)}`}>
                    {(metrics.reasoningAccuracy * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
              <Progress 
                value={metrics.reasoningAccuracy * 100} 
                className="h-2"
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-400">Adaptation Speed</span>
                <div className="flex items-center space-x-1">
                  {getTrendIcon(metrics.adaptationSpeed)}
                  <span className={`text-sm font-medium ${getMetricColor(metrics.adaptationSpeed)}`}>
                    {(metrics.adaptationSpeed * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
              <Progress 
                value={metrics.adaptationSpeed * 100} 
                className="h-2"
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-400">Self Consistency</span>
                <div className="flex items-center space-x-1">
                  {getTrendIcon(metrics.selfConsistency)}
                  <span className={`text-sm font-medium ${getMetricColor(metrics.selfConsistency)}`}>
                    {(metrics.selfConsistency * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
              <Progress 
                value={metrics.selfConsistency * 100} 
                className="h-2"
              />
            </div>
          </div>
        </div>

        {/* Meta-Cognitive State */}
        <div className="space-y-4">
          <h4 className="text-sm font-medium text-gray-300 flex items-center space-x-2">
            <Zap className="w-4 h-4" />
            <span>Meta-Cognitive State</span>
          </h4>
          
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400">Self Awareness</span>
              <Badge variant="outline" className={getMetricColor(metaCognitiveState.selfAwareness)}>
                {(metaCognitiveState.selfAwareness * 100).toFixed(0)}%
              </Badge>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400">Confidence</span>
              <Badge variant="outline" className={getMetricColor(metaCognitiveState.confidence)}>
                {(metaCognitiveState.confidence * 100).toFixed(0)}%
              </Badge>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400">Adaptation</span>
              <Badge variant="outline" className={getMetricColor(metaCognitiveState.adaptation)}>
                {(metaCognitiveState.adaptation * 100).toFixed(0)}%
              </Badge>
            </div>
          </div>

          {/* Reflection Display */}
          <div className="space-y-2">
            <span className="text-sm text-gray-400">Current Reflection</span>
            <div className="p-3 bg-gray-800/50 rounded-lg border border-gray-700">
              <p className="text-sm text-gray-300 italic">
                "{metaCognitiveState.reflection}"
              </p>
            </div>
          </div>
        </div>

        {/* Overall Status */}
        <div className="pt-4 border-t border-gray-700">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-gray-300">Overall Status</span>
            <div className="flex items-center space-x-2">
              <CheckCircle className="w-4 h-4 text-green-400" />
              <span className="text-sm text-green-400 font-medium">Conscious</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
