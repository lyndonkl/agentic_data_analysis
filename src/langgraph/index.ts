import { START, END, StateGraph } from '@langchain/langgraph';
import { StateAnnotation } from './state';
import { dataAnalyzer } from './nodes/analyzer';

export const createAnalysisGraph = () => {
  // Initialize the graph
  const workflow = new StateGraph(StateAnnotation)
    .addNode("analyzer", dataAnalyzer)
    .addEdge(START, "analyzer")
    .addEdge("analyzer", END)
    .compile();

  return workflow;
};

// Create and export the graph instance
const analysisGraph = createAnalysisGraph();

export { analysisGraph };
export * from './types';
export * from './state';
export * from './nodes/analyzer';