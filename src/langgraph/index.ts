import { START, END, StateGraph } from '@langchain/langgraph';
import { StateAnnotation } from './state';
import { dataAnalyzer } from './nodes/analyzer';
import { questionGenerator } from './nodes/question-generator';

export const createAnalysisGraph = () => {
  // Initialize the graph
  const workflow = new StateGraph(StateAnnotation)
    .addNode("analyzer", dataAnalyzer)
    .addNode("question_generator", questionGenerator)
    .addEdge(START, "analyzer")
    .addEdge("analyzer", "question_generator")
    .addEdge("question_generator", END)
    .compile();

  return workflow;
};

// Create and export the graph instance
const analysisGraph = createAnalysisGraph();

export { analysisGraph };
export * from './types';
export * from './state';
export * from './nodes/analyzer';
export * from './nodes/question-generator';