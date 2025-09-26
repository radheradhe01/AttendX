'use client';

import React, { useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { TrendingUp, Activity, Brain, Zap } from 'lucide-react';

interface PerformanceMetrics {
  memoryRetention: number;
  reasoningAccuracy: number;
  adaptationSpeed: number;
  selfConsistency: number;
}

interface PerformanceChartProps {
  performanceMetrics: PerformanceMetrics;
  lastUpdate: number;
}

export function PerformanceChart({ performanceMetrics, lastUpdate }: PerformanceChartProps) {
  const chartData = useMemo(() => {
    // Generate mock historical data for demonstration
    const now = Date.now();
    const data = [];
    
    for (let i = 29; i >= 0; i--) {
      const timestamp = now - (i * 60000); // 1 minute intervals
      const baseTime = i / 30;
      
      data.push({
        time: new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        timestamp,
        memoryRetention: Math.max(0.1, performanceMetrics.memoryRetention + (Math.sin(baseTime * Math.PI * 2) * 0.1) + (Math.random() - 0.5) * 0.05),
        reasoningAccuracy: Math.max(0.1, performanceMetrics.reasoningAccuracy + (Math.cos(baseTime * Math.PI * 2) * 0.1) + (Math.random() - 0.5) * 0.05),
        adaptationSpeed: Math.max(0.1, performanceMetrics.adaptationSpeed + (Math.sin(baseTime * Math.PI * 1.5) * 0.1) + (Math.random() - 0.5) * 0.05),
        selfConsistency: Math.max(0.1, performanceMetrics.selfConsistency + (Math.cos(baseTime * Math.PI * 1.5) * 0.1) + (Math.random() - 0.5) * 0.05),
      });
    }
    
    return data;
  }, [performanceMetrics]);

  const currentTrends = useMemo(() => {
    const recent = chartData.slice(-5);
    const older = chartData.slice(-10, -5);
    
    const trends = {
      memoryRetention: recent.reduce((sum, d) => sum + d.memoryRetention, 0) / 5 - 
                      older.reduce((sum, d) => sum + d.memoryRetention, 0) / 5,
      reasoningAccuracy: recent.reduce((sum, d) => sum + d.reasoningAccuracy, 0) / 5 - 
                        older.reduce((sum, d) => sum + d.reasoningAccuracy, 0) / 5,
      adaptationSpeed: recent.reduce((sum, d) => sum + d.adaptationSpeed, 0) / 5 - 
                      older.reduce((sum, d) => sum + d.adaptationSpeed, 0) / 5,
      selfConsistency: recent.reduce((sum, d) => sum + d.selfConsistency, 0) / 5 - 
                      older.reduce((sum, d) => sum + d.selfConsistency, 0) / 5,
    };
    
    return trends;
  }, [chartData]);

  const getTrendIcon = (trend: number) => {
    if (trend > 0.02) return <TrendingUp className="w-4 h-4 text-green-400" />;
    if (trend < -0.02) return <TrendingUp className="w-4 h-4 text-red-400 rotate-180" />;
    return <Activity className="w-4 h-4 text-gray-400" />;
  };

  const getTrendColor = (trend: number) => {
    if (trend > 0.02) return 'text-green-400';
    if (trend < -0.02) return 'text-red-400';
    return 'text-gray-400';
  };

  return (
    <Card className="consciousness-card">
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Brain className="w-5 h-5 text-consciousness-400" />
          <span>Performance Trends</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Current Trends */}
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400">Memory Retention</span>
              <div className="flex items-center space-x-1">
                {getTrendIcon(currentTrends.memoryRetention)}
                <span className={`text-sm font-medium ${getTrendColor(currentTrends.memoryRetention)}`}>
                  {(currentTrends.memoryRetention * 100).toFixed(1)}%
                </span>
              </div>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400">Reasoning Accuracy</span>
              <div className="flex items-center space-x-1">
                {getTrendIcon(currentTrends.reasoningAccuracy)}
                <span className={`text-sm font-medium ${getTrendColor(currentTrends.reasoningAccuracy)}`}>
                  {(currentTrends.reasoningAccuracy * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
          
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400">Adaptation Speed</span>
              <div className="flex items-center space-x-1">
                {getTrendIcon(currentTrends.adaptationSpeed)}
                <span className={`text-sm font-medium ${getTrendColor(currentTrends.adaptationSpeed)}`}>
                  {(currentTrends.adaptationSpeed * 100).toFixed(1)}%
                </span>
              </div>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400">Self Consistency</span>
              <div className="flex items-center space-x-1">
                {getTrendIcon(currentTrends.selfConsistency)}
                <span className={`text-sm font-medium ${getTrendColor(currentTrends.selfConsistency)}`}>
                  {(currentTrends.selfConsistency * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Performance Chart */}
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-gray-300 flex items-center space-x-2">
            <Zap className="w-4 h-4" />
            <span>30-Minute Performance History</span>
          </h4>
          
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="memoryGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                  </linearGradient>
                  <linearGradient id="reasoningGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
                  </linearGradient>
                  <linearGradient id="adaptationGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                  </linearGradient>
                  <linearGradient id="consistencyGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#f59e0b" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis 
                  dataKey="time" 
                  stroke="#9ca3af"
                  fontSize={12}
                  tickCount={6}
                />
                <YAxis 
                  stroke="#9ca3af"
                  fontSize={12}
                  domain={[0, 1]}
                  tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1f2937',
                    border: '1px solid #374151',
                    borderRadius: '8px',
                    color: '#f3f4f6'
                  }}
                  formatter={(value: number, name: string) => [
                    `${(value * 100).toFixed(1)}%`,
                    name.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())
                  ]}
                />
                
                <Area
                  type="monotone"
                  dataKey="memoryRetention"
                  stroke="#3b82f6"
                  fillOpacity={1}
                  fill="url(#memoryGradient)"
                  strokeWidth={2}
                />
                <Area
                  type="monotone"
                  dataKey="reasoningAccuracy"
                  stroke="#8b5cf6"
                  fillOpacity={1}
                  fill="url(#reasoningGradient)"
                  strokeWidth={2}
                />
                <Area
                  type="monotone"
                  dataKey="adaptationSpeed"
                  stroke="#10b981"
                  fillOpacity={1}
                  fill="url(#adaptationGradient)"
                  strokeWidth={2}
                />
                <Area
                  type="monotone"
                  dataKey="selfConsistency"
                  stroke="#f59e0b"
                  fillOpacity={1}
                  fill="url(#consistencyGradient)"
                  strokeWidth={2}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart Legend */}
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-blue-500 rounded-full" />
            <span className="text-gray-400">Memory Retention</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-purple-500 rounded-full" />
            <span className="text-gray-400">Reasoning Accuracy</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-green-500 rounded-full" />
            <span className="text-gray-400">Adaptation Speed</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-yellow-500 rounded-full" />
            <span className="text-gray-400">Self Consistency</span>
          </div>
        </div>

        {/* Last Update */}
        <div className="pt-4 border-t border-gray-700">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-gray-300">Last Update</span>
            <span className="text-sm text-gray-400">
              {new Date(lastUpdate).toLocaleTimeString()}
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
