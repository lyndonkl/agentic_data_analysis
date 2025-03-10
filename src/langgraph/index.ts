import { StateGraph } from '@langchain/langgraph';
import { BaseMessage } from '@langchain/core/messages';
import { RunnableConfig } from '@langchain/core/runnables';
import { GraphState } from './types';
import { initializeState } from './state';

export const createAnalysisGraph = () => {
  // Initialize the graph
  const workflow = new StateGraph<GraphState>({
    channels: ['analysis_result']
  });

  // Add nodes and edges will be implemented here
  
  // Compile the graph
  return workflow.compile();
};

export * from './types';
export * from './state';
export * from './nodes'; 