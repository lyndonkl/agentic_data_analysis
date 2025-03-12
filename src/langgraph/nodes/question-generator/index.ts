import { ChatOpenAI } from "@langchain/openai";
import { SystemMessage, HumanMessage, BaseMessage, AIMessage, ToolMessage } from "@langchain/core/messages";
import { StructuredOutputParser } from "@langchain/core/output_parsers";
import { GraphState, VisualizationQuestion, VisualizationQuestionSchema } from "../../types";
import { z } from 'zod';
import { v4 as uuidv4 } from 'uuid';
import { visualizationTools } from "../../tools/visualization";
import { type BaseMessageLike } from "@langchain/core/messages";
import { type ToolCall } from "@langchain/core/messages/tool";
import { addMessages } from "@langchain/langgraph";

// Tool argument types
interface SuggestGraphsArgs {
  numericCount: number;
  categoricalCount: number;
  numericOrdered: boolean;
  pointCount: 'few' | 'many';
}

type ToolArgs = SuggestGraphsArgs | Record<string, never>;

const outputParser = StructuredOutputParser.fromZodSchema(
  z.object({
    questions: z.array(VisualizationQuestionSchema.omit({ id: true }))
  })
);

const QUESTION_GENERATOR_PROMPT = `You are a Data Visualization Expert specializing in exploratory data analysis. Your task is to generate insightful questions that can be answered through data visualization.

For each question you generate, you must:
1. Focus only on the fields present in the dataset
2. Suggest questions that can be answered with basic charts (bar, line, scatter, pie, histogram)
3. Consider relationships between fields that might reveal interesting patterns
4. Prioritize questions that help understand distributions, trends, and relationships

Each question must be:
- Specific and focused on 1-2 fields
- Answerable through visual analysis
- Relevant to understanding the data's patterns
- Suitable for basic charting (no complex statistical analysis)

DO NOT:
- Suggest questions requiring fields not in the dataset
- Ask questions needing advanced statistical analysis
- Generate questions about predictions or future trends
- Include questions requiring data transformation

${outputParser.getFormatInstructions()}`;

// Create a map of tools by name for easy lookup
const toolsByName = Object.fromEntries(visualizationTools.map((tool) => [tool.name, tool]));

// Function to call a specific tool
async function callTool(toolCall: ToolCall): Promise<ToolMessage> {
  const tool = toolsByName[toolCall.name];
  if (!tool) {
    throw new Error(`Tool ${toolCall.name} not found`);
  }
  if (!toolCall.id) {
    throw new Error('Tool call ID is required');
  }
  
  const observation = await tool.invoke(toolCall);
  return new ToolMessage({ content: observation, tool_call_id: toolCall.id });
}

export async function questionGenerator(state: GraphState): Promise<Partial<GraphState>> {
  try {
    if (!state.metadata) {
      throw new Error("Metadata required for question generation");
    }

    const model = new ChatOpenAI({
      modelName: "gpt-4o",
      temperature: 0.7, // Higher temperature for more diverse questions
    }).bindTools(visualizationTools);

    const prompt = `Analyze this dataset and generate visualization questions:
Dataset Summary:
${state.metadata.summary}

Available Fields:
${Object.entries(state.metadata.fields).map(([name, meta]) => `
${name}:
- Type: ${meta.type}
- Description: ${meta.description}
${meta.range?.min !== undefined ? `- Range: ${meta.range.min} to ${meta.range.max}` : ''}
${meta.range?.uniqueValues ? `- Unique Values: ${meta.range.uniqueValues.length} different values` : ''}
`).join('\n')}

Generate at least 10 questions that can be answered through basic data visualizations. For each question:
1. Specify which visualization type is most appropriate (bar, line, scatter, pie, or histogram)
2. List the specific fields needed for the visualization
3. Explain why this visualization would be insightful

Focus on questions that:
- Explore distributions of numeric fields
- Compare categories
- Look for relationships between fields
- Analyze patterns in the data

You have access to two tools:
1. getGraphCatalog: Returns a list of all available graph types and when to use them
2. suggestGraphs: Suggests appropriate graph types based on the data structure`;

    let currentMessages: BaseMessageLike[] = [
      new SystemMessage(QUESTION_GENERATOR_PROMPT),
      new HumanMessage(prompt)
    ];

    // Initial model call with tools bound
    let llmResponse = await model.invoke(currentMessages);

    while (true) {
      if (!llmResponse.tool_calls?.length) {
        break;
      }

      // Execute tools
      const toolResults = await Promise.all(
        llmResponse.tool_calls.map((toolCall) => callTool(toolCall))
      );

      // Append to message list
      currentMessages = addMessages(currentMessages, [llmResponse, ...toolResults]);

      // Call model again
      llmResponse = await model.invoke(currentMessages);
    }

    // Parse the final response
    let parsedOutput;
    if (llmResponse instanceof BaseMessage) {
      parsedOutput = await outputParser.parse(llmResponse.content.toString());
    } else {
      throw new Error("Unexpected response format from model");
    }
    
    // Add UUIDs to questions
    const questions: VisualizationQuestion[] = parsedOutput.questions.map(q => ({
      id: uuidv4(),
      ...q
    }));

    // Update metadata with generated questions
    return {
      metadata: {
        fields: state.metadata.fields,
        rowCount: state.metadata.rowCount,
        summary: state.metadata.summary,
        dataQualityIssues: state.metadata.dataQualityIssues,
        questions
      }
    };
  } catch (error) {
    console.error("Error generating questions:", error);
    return {
      metadata: {
        fields: state.metadata?.fields ?? {},
        rowCount: state.metadata?.rowCount ?? 0,
        summary: state.metadata?.summary ?? 'Error generating questions',
        dataQualityIssues: state.metadata?.dataQualityIssues ?? [],
        questions: []
      }
    };
  }
} 