import { Annotation } from '@langchain/langgraph';
import { Data, DataSchema } from '../types';

export const StateAnnotation = Annotation.Root({
  data: Annotation<Data>({
    default: () => [],
    reducer: (current: Data, update: Data) => {
      // Merge current and update arrays to maintain state
      const merged = [...(current || []), ...update];
      return DataSchema.parse(merged);
    }
  })
});