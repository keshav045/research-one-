import React from 'react';
import { CheckCircle2, CircleDot, AlertCircle } from 'lucide-react';
import { PipelineStep as PipelineStepType, StepStatus } from '../../types';

interface PipelineStepProps {
  step: PipelineStepType;
  index: number;
}

export const PipelineStep: React.FC<PipelineStepProps> = ({ step, index }) => {
  const getStatusIndicator = (status: StepStatus) => {
    switch (status) {
      case 'completed':
        return <div className="w-6 h-6 rounded-full border-2 border-black bg-black flex items-center justify-center text-white shrink-0"><CheckCircle2 className="w-3.5 h-3.5" /></div>;
      case 'active':
        return <div className="w-6 h-6 rounded-full border-2 border-black bg-white flex items-center justify-center shrink-0"><CircleDot className="w-3.5 h-3.5 text-black animate-spin" /></div>;
      case 'failed':
        return <div className="w-6 h-6 rounded-full border-2 border-red-500 bg-red-50 flex items-center justify-center shrink-0"><AlertCircle className="w-3.5 h-3.5 text-red-500" /></div>;
      default:
        return <div className="w-6 h-6 rounded-full border-2 border-gray-200 bg-gray-50 flex items-center justify-center shrink-0"><span className="w-1.5 h-1.5 rounded-full bg-gray-300" /></div>;
    }
  };

  return (
    <div className="flex items-start gap-3 py-2.5">
      {getStatusIndicator(step.status)}
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between gap-2">
          <span className={`text-sm font-medium ${step.status === 'active' ? 'text-black' : step.status === 'completed' ? 'text-gray-900' : 'text-gray-400'}`}>
            <span className="text-gray-400 font-mono text-xs mr-1">0{index + 1}.</span>
            {step.name}
          </span>
          {step.timestamp && <span className="text-xs text-gray-400 font-mono">{step.timestamp}</span>}
        </div>
        <p className={`text-xs mt-0.5 leading-relaxed ${step.status === 'active' ? 'text-gray-700 font-medium' : 'text-gray-400'}`}>
          {step.description}
        </p>
      </div>
    </div>
  );
};
