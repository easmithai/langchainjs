import { CallableTool, Tool } from "@google/genai";
import { BindToolsInput } from "@langchain/core/language_models/chat_models";

export type GoogleGenerativeAIToolType =
  | BindToolsInput
  | Tool
  | CallableTool;
