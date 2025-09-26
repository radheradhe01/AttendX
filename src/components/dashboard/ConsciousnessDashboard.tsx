'use client';

import React from 'react';
import { useConsciousness } from '@/hooks/useConsciousness';
import { ConsciousnessMetrics } from './ConsciousnessMetrics';
import { MemoryOverview } from './MemoryOverview';
import { ReasoningOverview } from './ReasoningOverview';
import { AttentionOverview } from './AttentionOverview';
import { DecisionTimeline } from './DecisionTimeline';
import { PerformanceChart } from './PerformanceChart';
import { Brain, Database, Cpu, Network, TrendingUp, Clock } from 'lucide-react';

export function ConsciousnessDashboard() {
  const { consciousnessState } = useConsciousness();

  return (
    <div className="h-full overflow-y-auto scrollbar-hide">
      <div className="space-y-6">
        {/* Header */}
        <div className="text-center">
          <h1 className="text-4xl font-bold text-shadow-lg consciousness-gradient bg-clip-text text-transparent">
            Consciousness Dashboard
          </h1>
          <p className="text-gray-400 mt-2">
            Real-time monitoring of proto-conscious agent behavior
          </p>
        </div>

        {/* Status Overview */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="consciousness-card">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-consciousness-500/20 rounded-lg">
                <Brain className="w-6 h-6 text-consciousness-400" />
              </div>
              <div>
                <p className="text-sm text-gray-400">Consciousness Level</p>
                <p className="text-2xl font-bold text-consciousness-400">
                  {(consciousnessState.metaCognitiveState.selfAwareness * 100).toFixed(1)}%
                </p>
              </div>
            </div>
          </div>

          <div className="memory-card">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-memory-500/20 rounded-lg">
                <Database className="w-6 h-6 text-memory-400" />
              </div>
              <div>
                <p className="text-sm text-gray-400">Memory Traces</p>
                <p className="text-2xl font-bold text-memory-400">
                  {consciousnessState.memoryTraces.length}
                </p>
              </div>
            </div>
          </div>

          <div className="reasoning-card">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-reasoning-500/20 rounded-lg">
                <Cpu className="w-6 h-6 text-reasoning-400" />
              </div>
              <div>
                <p className="text-sm text-gray-400">Reasoning Steps</p>
                <p className="text-2xl font-bold text-reasoning-400">
                  {consciousnessState.reasoningSteps.length}
                </p>
              </div>
            </div>
          </div>

          <div className="attention-card">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-attention-500/20 rounded-lg">
                <Network className="w-6 h-6 text-attention-400" />
              </div>
              <div>
                <p className="text-sm text-gray-400">Attention Focus</p>
                <p className="text-2xl font-bold text-attention-400">
                  {consciousnessState.attentionFocus.length}
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column */}
          <div className="space-y-6">
            {/* Consciousness Metrics */}
            <ConsciousnessMetrics 
              metrics={consciousnessState.performanceMetrics}
              metaCognitiveState={consciousnessState.metaCognitiveState}
            />

            {/* Memory Overview */}
            <MemoryOverview 
              memoryTraces={consciousnessState.memoryTraces}
              performanceMetrics={consciousnessState.performanceMetrics}
            />

            {/* Reasoning Overview */}
            <ReasoningOverview 
              reasoningSteps={consciousnessState.reasoningSteps}
              performanceMetrics={consciousnessState.performanceMetrics}
            />
          </div>

          {/* Right Column */}
          <div className="space-y-6">
            {/* Attention Overview */}
            <AttentionOverview 
              attentionFocus={consciousnessState.attentionFocus}
              metaCognitiveState={consciousnessState.metaCognitiveState}
            />

            {/* Decision Timeline */}
            <DecisionTimeline 
              decisions={consciousnessState.decisionHistory}
              currentGoal={consciousnessState.currentGoal}
            />

            {/* Performance Chart */}
            <PerformanceChart 
              performanceMetrics={consciousnessState.performanceMetrics}
              lastUpdate={consciousnessState.lastUpdate}
            />
          </div>
        </div>

        {/* Current Goal and Status */}
        <div className="consciousness-card">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="p-3 bg-consciousness-500/20 rounded-lg">
                <TrendingUp className="w-8 h-8 text-consciousness-400" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-white">Current Goal</h3>
                <p className="text-gray-300">{consciousnessState.currentGoal}</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Clock className="w-5 h-5 text-gray-400" />
                <span className="text-sm text-gray-400">
                  Last update: {new Date(consciousnessState.lastUpdate).toLocaleTimeString()}
                </span>
              </div>
              <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                consciousnessState.isActive 
                  ? 'bg-green-500/20 text-green-400' 
                  : 'bg-gray-500/20 text-gray-400'
              }`}>
                {consciousnessState.isActive ? 'Active' : 'Inactive'}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
