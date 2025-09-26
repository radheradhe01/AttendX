'use client';

import { useState, useEffect } from 'react';
import { ConsciousnessDashboard } from '@/components/dashboard/ConsciousnessDashboard';
import { StreamOfConsciousness } from '@/components/consciousness/StreamOfConsciousness';
import { MemoryVisualization } from '@/components/memory/MemoryVisualization';
import { ReasoningVisualization } from '@/components/reasoning/ReasoningVisualization';
import { AttentionVisualization } from '@/components/attention/AttentionVisualization';
import { MetaCognitivePanel } from '@/components/meta/MetaCognitivePanel';
import { AgentControls } from '@/components/controls/AgentControls';
import { StatusBar } from '@/components/ui/StatusBar';
import { Sidebar } from '@/components/layout/Sidebar';
import { Header } from '@/components/layout/Header';
import { useConsciousness } from '@/hooks/useConsciousness';
import { Brain, Cpu, Database, Network, Zap } from 'lucide-react';

export default function HomePage() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [isAgentActive, setIsAgentActive] = useState(false);
  const { consciousnessState, startAgent, stopAgent, resetAgent } = useConsciousness();

  const tabs = [
    { id: 'dashboard', label: 'Dashboard', icon: Brain },
    { id: 'stream', label: 'Stream of Consciousness', icon: Zap },
    { id: 'memory', label: 'Memory Systems', icon: Database },
    { id: 'reasoning', label: 'Reasoning Engine', icon: Cpu },
    { id: 'attention', label: 'Attention Focus', icon: Network },
    { id: 'meta', label: 'Meta-Cognition', icon: Brain },
  ];

  const renderActiveTab = () => {
    switch (activeTab) {
      case 'dashboard':
        return <ConsciousnessDashboard />;
      case 'stream':
        return <StreamOfConsciousness />;
      case 'memory':
        return <MemoryVisualization />;
      case 'reasoning':
        return <ReasoningVisualization />;
      case 'attention':
        return <AttentionVisualization />;
      case 'meta':
        return <MetaCognitivePanel />;
      default:
        return <ConsciousnessDashboard />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 neural-network-bg">
      <div className="flex h-screen">
        {/* Sidebar */}
        <Sidebar 
          tabs={tabs} 
          activeTab={activeTab} 
          onTabChange={setActiveTab}
          className="w-64"
        />
        
        {/* Main Content */}
        <div className="flex-1 flex flex-col">
          {/* Header */}
          <Header 
            isAgentActive={isAgentActive}
            onToggleAgent={() => {
              if (isAgentActive) {
                stopAgent();
                setIsAgentActive(false);
              } else {
                startAgent();
                setIsAgentActive(true);
              }
            }}
            onResetAgent={() => {
              resetAgent();
              setIsAgentActive(false);
            }}
          />
          
          {/* Status Bar */}
          <StatusBar 
            consciousnessState={consciousnessState}
            isAgentActive={isAgentActive}
          />
          
          {/* Main Content Area */}
          <main className="flex-1 overflow-hidden">
            <div className="h-full p-6">
              {renderActiveTab()}
            </div>
          </main>
          
          {/* Agent Controls */}
          <div className="border-t border-gray-700 bg-gray-800/50 backdrop-blur-sm">
            <AgentControls 
              isAgentActive={isAgentActive}
              onStart={() => {
                startAgent();
                setIsAgentActive(true);
              }}
              onStop={() => {
                stopAgent();
                setIsAgentActive(false);
              }}
              onReset={() => {
                resetAgent();
                setIsAgentActive(false);
              }}
              consciousnessState={consciousnessState}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
