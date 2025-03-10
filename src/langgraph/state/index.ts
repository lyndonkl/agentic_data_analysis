import { Annotation } from '@langchain/langgraph';
import { Data, DataSchema, DatasetMetadata, DatasetMetadataSchema } from '../types';

export const StateAnnotation = Annotation.Root({
  data: Annotation<Data>({
    default: () => [],
    reducer: (current: Data, update: Data) => {
      // Merge current and update arrays to maintain state
      const merged = [...(current || []), ...update];
      return DataSchema.parse(merged);
    }
  }),
  metadata: Annotation<DatasetMetadata>({
    default: () => ({
      fields: {},
      rowCount: 0,
      summary: '',
      dataQualityIssues: []
    }),
    reducer: (_current, update) => DatasetMetadataSchema.parse(update)
  })
});