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
import { task } from "@langchain/langgraph";
import { Runnable } from "@langchain/core/runnables";
import { BaseLanguageModelInput } from "@langchain/core/language_models/base";
import { AIMessageChunk } from "@langchain/core/messages";

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

// Task for calling the model
const callModel = task(
  "callModel", 
  async (
    messages: BaseMessageLike[], 
    model: Runnable<BaseLanguageModelInput, AIMessageChunk>
  ) => {
    console.log("\n=== Calling Model ===");
    console.log("Input Messages:", JSON.stringify(messages.map(m => {
      if (m instanceof BaseMessage) {
        return {
          _type: m.constructor.name,
          content: m.content,
          kwargs: m.additional_kwargs
        };
      }
      return { _type: 'string', content: m };
    }), null, 2));
    
    const response = await model.invoke(messages);
    
    console.log("\nModel Response:", JSON.stringify({
      _type: response.constructor.name,
      content: response.content,
      tool_calls: response.tool_calls
    }, null, 2));
    
    return response;
  }
);

// Task for calling tools
const callTool = task(
  "callTool",
  async (toolCall: ToolCall): Promise<ToolMessage> => {
    console.log("\n=== Calling Tool ===");
    console.log("Tool Call:", JSON.stringify({
      name: toolCall.name,
      id: toolCall.id,
      args: toolCall.args
    }, null, 2));
    
    const tool = toolsByName[toolCall.name];
    if (!tool) {
      throw new Error(`Tool ${toolCall.name} not found`);
    }
    if (!toolCall.id) {
      throw new Error('Tool call ID is required');
    }
    
    const observation = await tool.invoke(toolCall);
    
    console.log("\nTool Response:", JSON.stringify({
      content: observation,
      tool_call_id: toolCall.id
    }, null, 2));
    
    const message = new ToolMessage({ content: observation, tool_call_id: toolCall.id });
    
    console.log("\nTool Message:", JSON.stringify({
      _type: message.constructor.name,
      content: message.content,
      kwargs: message.additional_kwargs
    }, null, 2));
    
    return message;
  }
);

export async function questionGenerator(state: GraphState): Promise<Partial<GraphState>> {
  try {
    if (!state.metadata) {
      throw new Error("Metadata required for question generation");
    }

    console.log("\n=== Starting Question Generator ===");
    
    const model = new ChatOpenAI({
      modelName: "gpt-4o",
      temperature: 0.7,
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

    console.log("\nInitial Messages:", JSON.stringify(currentMessages.map(m => {
      if (m instanceof BaseMessage) {
        return {
          _type: m.constructor.name,
          content: m.content?.toString().slice(0, 100) + "..." // Truncate for readability
        };
      }
      return { _type: 'string', content: m };
    }), null, 2));

    // Initial model call with tools bound
    let llmResponse = await callModel(currentMessages, model);

    while (true) {
      if (!llmResponse.tool_calls?.length) {
        console.log("\n=== No more tool calls, breaking loop ===");
        break;
      }

      console.log("\n=== Processing Tool Calls ===");
      console.log("Number of tool calls:", llmResponse.tool_calls.length);

      // Execute tools
      const toolResults = await Promise.all(
        llmResponse.tool_calls.map((toolCall) => callTool(toolCall))
      );

      console.log("\n=== Adding Messages to Context ===");
      console.log("Current message count:", currentMessages.length);
      console.log("Adding response and", toolResults.length, "tool results");

      // Append to message list
      currentMessages = addMessages(currentMessages, [llmResponse, ...toolResults]);

      console.log("New message count:", currentMessages.length);

      // Call model again
      llmResponse = await callModel(currentMessages, model);
    }

    console.log("\n=== Parsing Final Response ===");
    // Parse the final response
    let parsedOutput;
    if (llmResponse instanceof BaseMessage) {
      console.log("Response is BaseMessage, parsing content");
      parsedOutput = await outputParser.parse(llmResponse.content.toString());
    } else {
      console.log("Unexpected response type:", typeof llmResponse);
      throw new Error("Unexpected response format from model");
    }
    
    // Add UUIDs to questions
    const questions: VisualizationQuestion[] = parsedOutput.questions.map(q => ({
      id: uuidv4(),
      ...q
    }));

    console.log("\n=== Question Generation Complete ===");
    console.log("Generated", questions.length, "questions");

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
    console.error("\n=== Error in Question Generator ===");
    console.error("Error:", error);
    console.error("Stack:", error instanceof Error ? error.stack : "No stack trace");
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