import { ChatOpenAI } from "@langchain/openai";
import { BaseMessage } from "@langchain/core/messages";
import { SystemMessage, HumanMessage } from "@langchain/core/messages";
import { GraphState, FieldMetadata } from "../../types";

let globalState: GraphState;

interface NumericStats {
  min: number;
  max: number;
  mean: number;
  std: number;
}

const FIELD_ANALYSIS_PROMPT = `You are a seasoned Data Analyst specializing in data profiling and field analysis. Your task is to write clear, informative descriptions for each field in a dataset.

For each field, you will receive:
- Data type
- Basic statistics (for numeric fields)
- Value distribution (for categorical fields)
- Missing value counts
- Sample values

Create a concise but informative description that:
1. Explains what the field represents
2. Highlights key characteristics (range, distribution, patterns)
3. Notes any data quality concerns
4. Suggests potential uses for analysis

Keep descriptions factual and based on the provided statistics. Be specific about patterns you observe in the data.`;

const DATASET_SUMMARY_PROMPT = `You are a Data Scientist specializing in dataset analysis and understanding. Your task is to create a comprehensive summary of a dataset based on its field-level metadata and descriptions.

Consider:
1. Dataset Purpose and Content
   - What kind of data is this?
   - What entity or process does it describe?
   - What are the key fields and their relationships?

2. Data Quality Overview
   - Overall completeness of the data
   - Any systematic quality issues
   - Fields that may need attention

3. Analysis Potential
   - Key insights possible from this data
   - Relationships worth investigating
   - Potential use cases

Provide a clear, structured summary that helps analysts understand:
- What this dataset represents
- Its key characteristics and quality
- How it might be used effectively`;

const inspectField = (field: string) => {
  const values = globalState.data.map(item => item[field]).filter(x => x !== undefined);
  // Get the type from the first non-undefined value
  const type = values.length > 0 ? typeof values[0] : 'unknown';
  
  let stats: Partial<NumericStats> = {};
  let samples = [];
  let uniqueValues = undefined;

  if (type === 'number') {
    // For numeric fields: comprehensive statistics and samples
    const numbers = values as number[];
    const mean = numbers.reduce((a, b) => a + b, 0) / numbers.length;
    const variance = numbers.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / numbers.length;
    
    stats = {
      min: Math.min(...numbers),
      max: Math.max(...numbers),
      mean,
      std: Math.sqrt(variance)
    };
    samples = numbers
      .sort((a, b) => a - b)
      .filter((_, i, arr) => {
        // Get samples at 0%, 25%, 50%, 75%, 100% positions
        const positions = [0, 0.25, 0.5, 0.75, 1];
        return positions.some(p => i === Math.floor(p * (arr.length - 1)));
      });
  } else {
    // For categorical fields: get all unique values and their counts
    const valueMap = new Map();
    values.forEach(v => {
      const strVal = String(v);
      valueMap.set(strVal, (valueMap.get(strVal) || 0) + 1);
    });
    uniqueValues = Array.from(valueMap.entries()).map(([value, count]) => ({
      value,
      count,
      frequency: count / values.length
    }));
    samples = values.slice(0, 5);
  }

  return {
    type,
    samples,
    uniqueValues,
    totalValues: values.length,
    missingValues: globalState.data.length - values.length,
    stats
  };
};

async function generateFieldDescriptions(fields: Record<string, FieldMetadata>): Promise<Record<string, string>> {
  const model = new ChatOpenAI({
    modelName: "gpt-4o",
    temperature: 0.1,
  });

  const descriptions: Record<string, string> = {};
  
  for (const [fieldName, metadata] of Object.entries(fields)) {
    const prompt = `Analyze this field: "${fieldName}"

Field Statistics:
- Type: ${metadata.type}
- Total Values: ${metadata.totalCount}
- Missing Values: ${metadata.missingCount}
${metadata.range?.min !== undefined ? `- Range: ${metadata.range.min} to ${metadata.range.max}` : ''}
${metadata.range?.uniqueValues ? `- Unique Values: ${metadata.range.uniqueValues.length}` : ''}

Sample Values: ${metadata.examples?.join(', ')}
${metadata.range?.uniqueValues ? `\nValue Distribution: ${metadata.range.uniqueValues.slice(0, 10).join(', ')}${metadata.range.uniqueValues.length > 10 ? '...' : ''}` : ''}

Based on these statistics, provide a clear, concise description of what this field represents and its key characteristics.`;

    const response = await model.invoke([
      new SystemMessage(FIELD_ANALYSIS_PROMPT),
      new HumanMessage(prompt)
    ]);
    if (response instanceof BaseMessage) {
      descriptions[fieldName] = response.content.toString();
    } else {
      descriptions[fieldName] = `Field containing ${metadata.totalCount} values`;
    }
  }

  return descriptions;
}

async function generateDatasetSummary(
  fields: Record<string, FieldMetadata>, 
  rowCount: number
): Promise<string> {
  const model = new ChatOpenAI({
    modelName: "gpt-4o",
    temperature: 0.1,
  });

  const prompt = `Analyze this dataset with ${rowCount} records and the following fields:

${Object.entries(fields).map(([fieldName, metadata]) => `
### ${fieldName}
Type: ${metadata.type}
Description: ${metadata.description}
Values: ${metadata.totalCount} total, ${metadata.missingCount} missing
${metadata.range?.min !== undefined ? `Range: ${metadata.range.min} to ${metadata.range.max}` : ''}
${metadata.range?.uniqueValues ? `Unique Values: ${metadata.range.uniqueValues.length}` : ''}
`).join('\n')}

Based on these field descriptions and statistics, provide a comprehensive summary of:
1. What this dataset represents and its likely purpose
2. Key characteristics and patterns across fields
3. Overall data quality assessment
4. Potential analyses or insights possible with this data`;

  const response = await model.invoke([
    new SystemMessage(DATASET_SUMMARY_PROMPT),
    new HumanMessage(prompt)
  ]);

  return response instanceof BaseMessage ? response.content.toString() : 
    `Dataset with ${rowCount} records and ${Object.keys(fields).length} fields`;
}

export async function dataAnalyzer(state: GraphState): Promise<Partial<GraphState>> {
  try {
    globalState = state;
    
    const dataFields = Object.keys(state.data[0] || {});
    const metadataFields: Record<string, FieldMetadata> = {};

    // First pass: Gather all field statistics
    for (const field of dataFields) {
      const analysis = inspectField(field);
      
      metadataFields[field] = {
        type: analysis.type,
        description: '', // Placeholder for AI-generated description
        missingCount: analysis.missingValues,
        totalCount: analysis.totalValues,
        range: 'min' in analysis.stats ? {
          min: analysis.stats.min,
          max: analysis.stats.max,
          uniqueValues: analysis.uniqueValues?.map(v => v.value)
        } : {
          uniqueValues: analysis.uniqueValues?.map(v => v.value)
        },
        examples: analysis.samples
      };
    }
    
    console.log(metadataFields);

    // Second pass: Generate AI descriptions for each field
    const descriptions = await generateFieldDescriptions(metadataFields);
    
    // Update metadata with AI-generated descriptions
    for (const [field, description] of Object.entries(descriptions)) {
      metadataFields[field].description = description;
    }

    // Third pass: Generate overall dataset summary
    const summary = await generateDatasetSummary(metadataFields, state.data.length);

    console.log(metadataFields);

    return {
      metadata: {
        fields: metadataFields,
        rowCount: state.data.length,
        summary,
        dataQualityIssues: Object.entries(metadataFields)
          .filter(([_, meta]) => meta.missingCount > 0)
          .map(([field, meta]) => 
            `${field}: ${meta.missingCount} missing values out of ${meta.totalCount}`)
      }
    };
  } catch (error) {
    console.error("Error analyzing data:", error);
    return {
      metadata: {
        fields: {},
        rowCount: 0,
        summary: `Error during analysis: ${error instanceof Error ? error.message : 'Unknown error'}`,
        dataQualityIssues: ["Analysis failed"]
      }
    };
  }
} 