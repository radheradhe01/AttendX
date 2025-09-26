'use client';

import React, { useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Cpu, Clock, CheckCircle, AlertCircle, Brain } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';

interface ReasoningStep {
  id: string;
  rule: string;
  confidence: number;
  timestamp: number;
  explanation: string;
}

interface PerformanceMetrics {
  memoryRetention: number;
  reasoningAccuracy: number;
  adaptationSpeed: number;
  selfConsistency: number;
}

interface ReasoningOverviewProps {
  reasoningSteps: ReasoningStep[];
  performanceMetrics: PerformanceMetrics;
}

export function ReasoningOverview({ reasoningSteps, performanceMetrics }: ReasoningOverviewProps) {
  const reasoningStats = useMemo(() => {
    const highConfidence = reasoningSteps.filter(step => step.confidence >= 0.8).length;
    const mediumConfidence = reasoningSteps.filter(step => step.confidence >= 0.5 && step.confidence < 0.8).length;
    const lowConfidence = reasoningSteps.filter(step => step.confidence < 0.5).length;
    
    const avgConfidence = reasoningSteps.length > 0 
      ? reasoningSteps.reduce((sum, step) => sum + step.confidence, 0) / reasoningSteps.length 
      : 0;

    const recentSteps = reasoningSteps
      .sort((a, b) => b.timestamp - a.timestamp)
      .slice(0, 5);

    const ruleFrequency = reasoningSteps.reduce((acc, step) => {
      acc[step.rule] = (acc[step.rule] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    const mostUsedRule = Object.entries(ruleFrequency)
      .sort(([,a], [,b]) => b - a)[0];

    return {
      total: reasoningSteps.length,
      highConfidence,
      mediumConfidence,
      lowConfidence,
      avgConfidence,
      recentSteps,
      mostUsedRule: mostUsedRule ? mostUsedRule[0] : 'None'
    };
  }, [reasoningSteps]);

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-400';
    if (confidence >= 0.5) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getConfidenceIcon = (confidence: number) => {
    if (confidence >= 0.8) return <CheckCircle className="w-4 h-4 text-green-400" />;
    if (confidence >= 0.5) return <AlertCircle className="w-4 h-4 text-yellow-400" />;
    return <AlertCircle className="w-4 h-4 text-red-400" />;
  };

  return (
    <Card className="reasoning-card">
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Cpu className="w-5 h-5 text-reasoning-400" />
          <span>Reasoning Overview</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Reasoning Statistics */}
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400">Total Steps</span>
              <span className="text-lg font-bold text-reasoning-400">{reasoningStats.total}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400">Accuracy</span>
              <span className="text-lg font-bold text-green-400">
                {(performanceMetrics.reasoningAccuracy * 100).toFixed(0)}%
              </span>
            </div>
          </div>
          
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400">Avg Confidence</span>
              <span className={`text-lg font-bold ${getConfidenceColor(reasoningStats.avgConfidence)}`}>
                {(reasoningStats.avgConfidence * 100).toFixed(0)}%
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400">Most Used Rule</span>
              <span className="text-sm font-medium text-gray-300 truncate max-w-24">
                {reasoningStats.mostUsedRule}
              </span>
            </div>
          </div>
        </div>

        {/* Confidence Distribution */}
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-gray-300">Confidence Distribution</h4>
          
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <CheckCircle className="w-4 h-4 text-green-400" />
                <span className="text-sm text-gray-400">High (≥80%)</span>
              </div>
              <span className="text-sm font-medium text-green-400">
                {reasoningStats.highConfidence}
              </span>
            </div>
            
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <AlertCircle className="w-4 h-4 text-yellow-400" />
                <span className="text-sm text-gray-400">Medium (50-79%)</span>
              </div>
              <span className="text-sm font-medium text-yellow-400">
                {reasoningStats.mediumConfidence}
              </span>
            </div>
            
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <AlertCircle className="w-4 h-4 text-red-400" />
                <span className="text-sm text-gray-400">Low (&lt;50%)</span>
              </div>
              <span className="text-sm font-medium text-red-400">
                {reasoningStats.lowConfidence}
              </span>
            </div>
          </div>
        </div>

        {/* Recent Reasoning Steps */}
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-gray-300 flex items-center space-x-2">
            <Clock className="w-4 h-4" />
            <span>Recent Steps</span>
          </h4>
          
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {reasoningStats.recentSteps.length > 0 ? (
              reasoningStats.recentSteps.map((step) => (
                <div 
                  key={step.id}
                  className="p-3 bg-gray-800/30 rounded-lg border border-gray-700/50 hover:border-gray-600/50 transition-colors"
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      {getConfidenceIcon(step.confidence)}
                      <Badge variant="outline" className="text-xs">
                        {step.rule}
                      </Badge>
                    </div>
                    <span className="text-xs text-gray-500">
                      {formatDistanceToNow(new Date(step.timestamp), { addSuffix: true })}
                    </span>
                  </div>
                  
                  <p className="text-sm text-gray-300 mb-2 line-clamp-2">
                    {step.explanation}
                  </p>
                  
                  <div className="flex items-center justify-between text-xs text-gray-500">
                    <span>Confidence: {(step.confidence * 100).toFixed(0)}%</span>
                    <span className={getConfidenceColor(step.confidence)}>
                      {step.confidence >= 0.8 ? 'High' : step.confidence >= 0.5 ? 'Medium' : 'Low'}
                    </span>
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center py-8 text-gray-500">
                <Brain className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p className="text-sm">No reasoning steps yet</p>
              </div>
            )}
          </div>
        </div>

        {/* Reasoning Quality Indicator */}
        <div className="pt-4 border-t border-gray-700">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-gray-300">Reasoning Quality</span>
            <div className="flex items-center space-x-2">
              {performanceMetrics.reasoningAccuracy >= 0.8 ? (
                <>
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  <span className="text-sm text-green-400 font-medium">Excellent</span>
                </>
              ) : performanceMetrics.reasoningAccuracy >= 0.6 ? (
                <>
                  <AlertCircle className="w-4 h-4 text-yellow-400" />
                  <span className="text-sm text-yellow-400 font-medium">Good</span>
                </>
              ) : (
                <>
                  <AlertCircle className="w-4 h-4 text-red-400" />
                  <span className="text-sm text-red-400 font-medium">Needs Improvement</span>
                </>
              )}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
