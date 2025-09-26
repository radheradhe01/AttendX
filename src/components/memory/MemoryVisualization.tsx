'use client';

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Database, Brain } from 'lucide-react';

export function MemoryVisualization() {
  return (
    <div className="h-full flex flex-col">
      <Card className="flex-1 memory-card">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Database className="w-5 h-5 text-memory-400" />
            <span>Memory Visualization</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="flex-1 flex items-center justify-center">
          <div className="text-center text-gray-500">
            <Brain className="w-16 h-16 mx-auto mb-4 opacity-50" />
            <p className="text-lg font-medium mb-2">Memory Visualization</p>
            <p className="text-sm">Advanced memory visualization features coming soon</p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
