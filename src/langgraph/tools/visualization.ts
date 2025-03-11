import { ChatOpenAI } from "@langchain/openai";
import { tool } from "@langchain/core/tools";
import { z } from "zod";

// Graph type interface
export interface GraphType {
  key: string;
  description: string;
}

// The catalog of available graph types
const graphCatalog: GraphType[] = [
  {
    key: "histogram",
    description: "Use for 1 numeric variable with MANY data points (not necessarily ordered). Shows distribution."
  },
  {
    key: "density-plot",
    description: "Use for 1 numeric variable with MANY data points to visualize the smooth distribution shape."
  },
  {
    key: "boxplot",
    description: "Use for 1+ numeric variables (often grouped by category) to compare distributions, outliers, median, etc."
  },
  {
    key: "violin-plot",
    description: "Similar to a boxplot but shows the full distribution shape. Useful for 1+ numeric variables, especially grouped."
  },
  {
    key: "lollipop-plot",
    description: "Good for FEW data points, often along a categorical axis (1 numeric + 1 categorical). Emphasizes differences."
  },
  {
    key: "barplot",
    description: "Displays numeric values aggregated by 1 or more categorical variables (counts, sums, etc.)."
  },
  {
    key: "scatterplot",
    description: "Use for 2 numeric variables that are NOT ordered. Shows relationship, correlation, clustering."
  },
  {
    key: "connected-scatterplot",
    description: "Use for 2 numeric variables that ARE ordered (e.g. time or a sequence). Points connected in order."
  },
  {
    key: "area-plot",
    description: "Use for numeric trends over an ORDERED axis (often time). Fills area under the line."
  },
  {
    key: "stacked-area",
    description: "Visualizes multiple numeric series over an ORDERED axis by stacking their areas."
  },
  {
    key: "streamgraph",
    description: "Variation of stacked area for multiple ordered numeric series, creating a flowing shape."
  },
  {
    key: "heatmap",
    description: "Uses color to represent numeric values across 2D space (numeric vs. numeric or numeric vs. category)."
  },
  {
    key: "treemap",
    description: "Represents hierarchical or nested categorical data using nested rectangles sized by a numeric value."
  },
  {
    key: "venn-diagram",
    description: "Shows overlaps among 2+ categorical sets."
  },
  {
    key: "grouped-scatter",
    description: "Scatterplot points grouped by categories, can show subgroups in scatter form."
  },
  {
    key: "network",
    description: "Visualizes relationships (edges) between entities (nodes)."
  },
  {
    key: "dendrogram",
    description: "Shows hierarchical relationships in a tree structure."
  },
  {
    key: "hierarchical-edge-bundling",
    description: "Another approach to visualize hierarchical data with edges bundled to reduce clutter."
  },
  {
    key: "line-chart",
    description: "Typically used for time series (1 numeric variable over an ordered axis). Points connected in chronological order."
  }
];

// Tool 1: Get Graph Catalog
export const getGraphCatalog = tool(
  async () => {
    return JSON.stringify(graphCatalog, null, 2);
  },
  {
    name: "getGraphCatalog",
    description: "Returns a comprehensive catalog of available graph types and when to use them.",
    schema: z.object({})
  }
);

// Tool 2: Suggest Graphs
export const suggestGraphs = tool(
  async ({ numericCount, categoricalCount, numericOrdered, pointCount }) => {
    const suggestions: string[] = [];

    // CASE A: Only numeric variables
    if (categoricalCount === 0) {
      if (numericCount === 1) {
        if (pointCount === "few") {
          suggestions.push("lollipop-plot", "boxplot");
        } else {
          suggestions.push("histogram", "density-plot", "boxplot", "violin-plot");
        }
      } else if (numericCount === 2) {
        if (numericOrdered) {
          suggestions.push("connected-scatterplot", "area-plot", "line-chart");
        } else {
          suggestions.push("scatterplot");
        }
      } else if (numericCount >= 3) {
        if (numericOrdered) {
          suggestions.push("stacked-area", "streamgraph", "line-chart");
        } else {
          suggestions.push("heatmap", "scatterplot");
        }
      }
    }
    // CASE B: Only categorical variables
    else if (numericCount === 0) {
      if (categoricalCount === 1) {
        suggestions.push("barplot");
        if (pointCount === "few") {
          suggestions.push("lollipop-plot");
        }
      } else {
        suggestions.push("venn-diagram", "treemap", "grouped-scatter", "network");
      }
    }
    // CASE C: Mixed numeric + categorical
    else {
      if (numericCount === 1 && categoricalCount === 1) {
        if (pointCount === "few") {
          suggestions.push("lollipop-plot", "barplot");
        } else {
          suggestions.push("boxplot", "violin-plot", "barplot");
        }
      } else {
        suggestions.push("barplot", "boxplot", "violin-plot", "heatmap");
        if (numericOrdered) {
          suggestions.push("connected-scatterplot", "area-plot");
        }
      }
    }

    // Find the full descriptions for the suggested graph types
    const suggestedGraphs = graphCatalog
      .filter(graph => suggestions.includes(graph.key))
      .map(graph => ({
        key: graph.key,
        description: graph.description
      }));

    return JSON.stringify(suggestedGraphs, null, 2);
  },
  {
    name: "suggestGraphs",
    description: "Suggests appropriate graph types based on the data structure (number of numeric/categorical variables, ordering, and point count).",
    schema: z.object({
      numericCount: z.number().describe("Number of numeric variables in consideration"),
      categoricalCount: z.number().describe("Number of categorical variables in consideration"),
      numericOrdered: z.boolean().describe("Whether the numeric variable(s) represent an ordered dimension (e.g. time/sequence)"),
      pointCount: z.enum(["few", "many"]).describe("Whether there are few or many data points")
    })
  }
);

// Export the tools array
export const visualizationTools = [getGraphCatalog, suggestGraphs]; 