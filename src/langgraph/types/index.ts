import { z } from 'zod';

// Define the schema for field metadata
export const FieldMetadataSchema = z.object({
  type: z.string(),
  description: z.string(),
  range: z.object({
    min: z.number().optional(),
    max: z.number().optional(),
    uniqueValues: z.array(z.unknown()).optional()
  }).optional(),
  missingCount: z.number(),
  totalCount: z.number(),
  examples: z.array(z.unknown()).optional()
});

export const VisualizationQuestionSchema = z.object({
  id: z.string(),
  question: z.string(),
  type: z.enum(['histogram', 'density-plot', 'boxplot', 'violin-plot', 'lollipop-plot', 'barplot', 'scatterplot', 'connected-scatterplot', 'area-plot', 'stacked-area', 'streamgraph', 'heatmap', 'treemap', 'venn-diagram', 'grouped-scatter', 'network', 'dendrogram', 'hierarchical-edge-bundling', 'line-chart']),
  fields: z.array(z.string()),
  description: z.string()
});

// Define the schema for dataset metadata
export const DatasetMetadataSchema = z.object({
  fields: z.record(z.string(), FieldMetadataSchema),
  rowCount: z.number(),
  summary: z.string(),
  dataQualityIssues: z.array(z.string()).optional(),
  questions: z.array(VisualizationQuestionSchema).optional()
});

// Define the data schema for JSON array of objects
export const DataSchema = z.array(z.record(z.string(), z.unknown()));

// TypeScript types for the data
export type Data = z.infer<typeof DataSchema>;
export type FieldMetadata = z.infer<typeof FieldMetadataSchema>;
export type DatasetMetadata = z.infer<typeof DatasetMetadataSchema>;
export type VisualizationQuestion = z.infer<typeof VisualizationQuestionSchema>;

// Graph state interface
export interface GraphState {
  data: Data;
  metadata?: DatasetMetadata;
}