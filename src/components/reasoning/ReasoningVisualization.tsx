'use client';

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Cpu, Brain } from 'lucide-react';

export function ReasoningVisualization() {
  return (
    <div className="h-full flex flex-col">
      <Card className="flex-1 reasoning-card">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Cpu className="w-5 h-5 text-reasoning-400" />
            <span>Reasoning Visualization</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="flex-1 flex items-center justify-center">
          <div className="text-center text-gray-500">
            <Brain className="w-16 h-16 mx-auto mb-4 opacity-50" />
            <p className="text-lg font-medium mb-2">Reasoning Visualization</p>
            <p className="text-sm">Advanced reasoning visualization features coming soon</p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
