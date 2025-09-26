'use client';

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Network, Brain } from 'lucide-react';

export function AttentionVisualization() {
  return (
    <div className="h-full flex flex-col">
      <Card className="flex-1 attention-card">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Network className="w-5 h-5 text-attention-400" />
            <span>Attention Visualization</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="flex-1 flex items-center justify-center">
          <div className="text-center text-gray-500">
            <Brain className="w-16 h-16 mx-auto mb-4 opacity-50" />
            <p className="text-lg font-medium mb-2">Attention Visualization</p>
            <p className="text-sm">Advanced attention visualization features coming soon</p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
