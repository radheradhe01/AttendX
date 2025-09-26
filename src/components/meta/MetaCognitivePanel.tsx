'use client';

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Brain } from 'lucide-react';

export function MetaCognitivePanel() {
  return (
    <div className="h-full flex flex-col">
      <Card className="flex-1 consciousness-card">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Brain className="w-5 h-5 text-consciousness-400" />
            <span>Meta-Cognitive Panel</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="flex-1 flex items-center justify-center">
          <div className="text-center text-gray-500">
            <Brain className="w-16 h-16 mx-auto mb-4 opacity-50" />
            <p className="text-lg font-medium mb-2">Meta-Cognitive Panel</p>
            <p className="text-sm">Advanced meta-cognitive visualization features coming soon</p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
