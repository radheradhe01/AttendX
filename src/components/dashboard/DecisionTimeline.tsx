'use client';

import React, { useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Clock, Target, TrendingUp, CheckCircle, AlertCircle, XCircle } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';

interface Decision {
  id: string;
  action: string;
  horizon: 'short' | 'medium' | 'long';
  confidence: number;
  reasoning: string;
  timestamp: number;
  outcome?: string;
}

interface DecisionTimelineProps {
  decisions: Decision[];
  currentGoal: string;
}

export function DecisionTimeline({ decisions, currentGoal }: DecisionTimelineProps) {
  const decisionStats = useMemo(() => {
    const shortTerm = decisions.filter(d => d.horizon === 'short').length;
    const mediumTerm = decisions.filter(d => d.horizon === 'medium').length;
    const longTerm = decisions.filter(d => d.horizon === 'long').length;
    
    const avgConfidence = decisions.length > 0 
      ? decisions.reduce((sum, d) => sum + d.confidence, 0) / decisions.length 
      : 0;

    const recentDecisions = decisions
      .sort((a, b) => b.timestamp - a.timestamp)
      .slice(0, 8);

    const successfulDecisions = decisions.filter(d => d.outcome === 'success').length;
    const successRate = decisions.length > 0 ? successfulDecisions / decisions.length : 0;

    return {
      total: decisions.length,
      shortTerm,
      mediumTerm,
      longTerm,
      avgConfidence,
      recentDecisions,
      successRate
    };
  }, [decisions]);

  const getHorizonColor = (horizon: 'short' | 'medium' | 'long') => {
    switch (horizon) {
      case 'short': return 'bg-green-500/20 text-green-400';
      case 'medium': return 'bg-yellow-500/20 text-yellow-400';
      case 'long': return 'bg-blue-500/20 text-blue-400';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-400';
    if (confidence >= 0.5) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getConfidenceIcon = (confidence: number) => {
    if (confidence >= 0.8) return <CheckCircle className="w-4 h-4 text-green-400" />;
    if (confidence >= 0.5) return <AlertCircle className="w-4 h-4 text-yellow-400" />;
    return <XCircle className="w-4 h-4 text-red-400" />;
  };

  const getOutcomeIcon = (outcome?: string) => {
    if (outcome === 'success') return <CheckCircle className="w-4 h-4 text-green-400" />;
    if (outcome === 'failure') return <XCircle className="w-4 h-4 text-red-400" />;
    return <AlertCircle className="w-4 h-4 text-gray-400" />;
  };

  return (
    <Card className="consciousness-card">
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Clock className="w-5 h-5 text-consciousness-400" />
          <span>Decision Timeline</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Current Goal */}
        <div className="p-4 bg-gray-800/50 rounded-lg border border-gray-700">
          <div className="flex items-center space-x-2 mb-2">
            <Target className="w-4 h-4 text-consciousness-400" />
            <span className="text-sm font-medium text-gray-300">Current Goal</span>
          </div>
          <p className="text-sm text-gray-300">{currentGoal}</p>
        </div>

        {/* Decision Statistics */}
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400">Total Decisions</span>
              <span className="text-lg font-bold text-consciousness-400">{decisionStats.total}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400">Success Rate</span>
              <span className="text-lg font-bold text-green-400">
                {(decisionStats.successRate * 100).toFixed(0)}%
              </span>
            </div>
          </div>
          
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400">Avg Confidence</span>
              <span className={`text-lg font-bold ${getConfidenceColor(decisionStats.avgConfidence)}`}>
                {(decisionStats.avgConfidence * 100).toFixed(0)}%
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400">Horizon Distribution</span>
              <div className="flex space-x-1">
                <Badge className="bg-green-500/20 text-green-400 text-xs">
                  {decisionStats.shortTerm}S
                </Badge>
                <Badge className="bg-yellow-500/20 text-yellow-400 text-xs">
                  {decisionStats.mediumTerm}M
                </Badge>
                <Badge className="bg-blue-500/20 text-blue-400 text-xs">
                  {decisionStats.longTerm}L
                </Badge>
              </div>
            </div>
          </div>
        </div>

        {/* Recent Decisions Timeline */}
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-gray-300 flex items-center space-x-2">
            <TrendingUp className="w-4 h-4" />
            <span>Recent Decisions</span>
          </h4>
          
          <div className="space-y-3 max-h-64 overflow-y-auto">
            {decisionStats.recentDecisions.length > 0 ? (
              decisionStats.recentDecisions.map((decision, index) => (
                <div 
                  key={decision.id}
                  className="flex items-start space-x-3 p-3 bg-gray-800/30 rounded-lg border border-gray-700/50 hover:border-gray-600/50 transition-colors"
                >
                  {/* Timeline indicator */}
                  <div className="flex flex-col items-center">
                    <div className={`w-3 h-3 rounded-full ${
                      decision.confidence >= 0.8 ? 'bg-green-400' : 
                      decision.confidence >= 0.5 ? 'bg-yellow-400' : 'bg-red-400'
                    }`} />
                    {index < decisionStats.recentDecisions.length - 1 && (
                      <div className="w-px h-8 bg-gray-600 mt-2" />
                    )}
                  </div>
                  
                  {/* Decision content */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <Badge className={getHorizonColor(decision.horizon)}>
                          {decision.horizon}
                        </Badge>
                        {getConfidenceIcon(decision.confidence)}
                        {decision.outcome && getOutcomeIcon(decision.outcome)}
                      </div>
                      <span className="text-xs text-gray-500">
                        {formatDistanceToNow(new Date(decision.timestamp), { addSuffix: true })}
                      </span>
                    </div>
                    
                    <p className="text-sm font-medium text-gray-300 mb-1">
                      {decision.action}
                    </p>
                    
                    <p className="text-xs text-gray-400 mb-2 line-clamp-2">
                      {decision.reasoning}
                    </p>
                    
                    <div className="flex items-center justify-between text-xs text-gray-500">
                      <span>Confidence: {(decision.confidence * 100).toFixed(0)}%</span>
                      {decision.outcome && (
                        <span className={decision.outcome === 'success' ? 'text-green-400' : 'text-red-400'}>
                          {decision.outcome}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center py-8 text-gray-500">
                <Clock className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p className="text-sm">No decisions yet</p>
              </div>
            )}
          </div>
        </div>

        {/* Decision Quality Indicator */}
        <div className="pt-4 border-t border-gray-700">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-gray-300">Decision Quality</span>
            <div className="flex items-center space-x-2">
              {decisionStats.successRate >= 0.8 ? (
                <>
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  <span className="text-sm text-green-400 font-medium">Excellent</span>
                </>
              ) : decisionStats.successRate >= 0.6 ? (
                <>
                  <AlertCircle className="w-4 h-4 text-yellow-400" />
                  <span className="text-sm text-yellow-400 font-medium">Good</span>
                </>
              ) : (
                <>
                  <XCircle className="w-4 h-4 text-red-400" />
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
