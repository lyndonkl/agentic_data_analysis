import { RunnableSequence } from '@langchain/core/runnables';
import { ChatOpenAI } from '@langchain/openai';
import { GraphState, NodeConfig } from '../types';

// Base configuration for nodes
export const createBaseNode = (config: NodeConfig) => {
  const model = new ChatOpenAI({
    modelName: 'gpt-4-turbo-preview',
    temperature: 0
  });

  return RunnableSequence.from([
    // Add node-specific logic here
    model
  ]);
};

// Export specific nodes as they are implemented
export * from './analyzer';
export * from './processor'; // Will be implemented 