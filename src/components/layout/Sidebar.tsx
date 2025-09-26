'use client';

import React from 'react';
import { cn } from '@/lib/utils';
import { LucideIcon } from 'lucide-react';

interface Tab {
  id: string;
  label: string;
  icon: LucideIcon;
}

interface SidebarProps {
  tabs: Tab[];
  activeTab: string;
  onTabChange: (tabId: string) => void;
  className?: string;
}

export function Sidebar({ tabs, activeTab, onTabChange, className }: SidebarProps) {
  return (
    <div className={cn(
      "bg-gray-900/95 backdrop-blur-sm border-r border-gray-700 flex flex-col",
      className
    )}>
      {/* Logo/Title */}
      <div className="p-6 border-b border-gray-700">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-gradient-to-br from-consciousness-400 to-consciousness-600 rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-sm">AC</span>
          </div>
          <div>
            <h1 className="text-lg font-bold text-white">Consciousness</h1>
            <p className="text-xs text-gray-400">Simulator</p>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <nav className="flex-1 p-4 space-y-2">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          
          return (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              className={cn(
                "w-full flex items-center space-x-3 px-4 py-3 rounded-lg text-left transition-all duration-200",
                isActive
                  ? "bg-consciousness-500/20 text-consciousness-400 border border-consciousness-500/30"
                  : "text-gray-400 hover:text-gray-300 hover:bg-gray-800/50"
              )}
            >
              <Icon className={cn(
                "w-5 h-5",
                isActive ? "text-consciousness-400" : "text-gray-500"
              )} />
              <span className="font-medium">{tab.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-gray-700">
        <div className="text-xs text-gray-500 text-center">
          <p>Proto-Conscious Agent</p>
          <p>v1.0.0</p>
        </div>
      </div>
    </div>
  );
}
